"""Aggregate comparable IFP ablations across several seeds."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from statistics import fmean, stdev
from tempfile import NamedTemporaryFile
from typing import Any

from .ifp_results import (
    ClosedLoopResult,
    IFPAblationError,
    IFP_ABLATION_SCHEMA,
    IFP_PROTOCOL_LIMITATION,
)
from .ifp_preregistration import (
    IFP_PREREGISTRATION_CLAIM_SCOPE,
    IFP_VARIANT_SET as IFP_ABLATION_STEPS,
    IFPPreregistration,
    IFPPreregistrationError,
    evaluate_ifp_endpoints,
    load_ifp_preregistration_bytes,
    normalize_optimizer_protocol,
)
from .model import ActionDecoder


IFP_STUDY_SCHEMA = "so101_wam.ifp_study.v4"
MIN_STUDY_SEEDS = 3
_REQUIRED_COMPARABILITY = (
    "same_closed_loop_protocol",
    "same_initial_model_state",
    "same_optimizer_budget",
    "same_seed",
    "same_train_split",
    "same_training_schedule",
    "same_validation_split",
    "same_window_count",
)
_RESULT_FIELDS = {
    "artifact",
    "artifact_sha256",
    "execution_failure_count",
    "failure_counts",
    "scored_trial_count",
    "success_count",
    "success_rate",
    "terminal_error_mean",
}


class IFPStudyError(ValueError):
    """Raised when seed reports cannot form one comparable study."""


def aggregate_ifp_study(
    report_paths: Sequence[str | Path],
    *,
    report_path: str | Path,
    preregistration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, summarize, and immutably publish multi-seed evidence."""

    output = Path(report_path).resolve()
    if output.exists():
        raise IFPStudyError(f"IFP study report already exists: {output}")
    preregistration, preregistration_sha256 = _load_preregistration(
        preregistration_path,
        output=output,
    )
    paths = tuple(Path(path) for path in report_paths)
    if len(paths) < MIN_STUDY_SEEDS:
        raise IFPStudyError(
            f"IFP study requires at least {MIN_STUDY_SEEDS} seed reports"
        )

    root = output.parent.resolve()
    sources = [_load_source(path, root=root, output=output) for path in paths]
    seeds = [source["seed"] for source in sources]
    if len(set(seeds)) != len(seeds):
        raise IFPStudyError("IFP study requires unique seeds")
    sources.sort(key=lambda source: source["seed"])

    _verify_study_contract(sources)
    if preregistration is not None:
        _verify_preregistration(preregistration, sources)
    variants = [
        _variant_summary(ifp_steps, sources)
        for ifp_steps in IFP_ABLATION_STEPS
    ]
    if preregistration is None:
        scientific_claim = "not_evaluated"
        result_semantics = "descriptive multi-seed K=0/2/4 summary"
    else:
        scientific_claim = "local_ifp_diagnostic_evaluated"
        result_semantics = "local plan-bound K=4 versus K=0 diagnostic"
    study = {
        "schema_version": IFP_STUDY_SCHEMA,
        "result": "complete",
        "evidence_level": "offline_and_simulation_diagnostic",
        "robot_used": False,
        "scientific_claim": scientific_claim,
        "result_semantics": result_semantics,
        "seed_count": len(sources),
        "seeds": [source["seed"] for source in sources],
        "comparability": {
            "minimum_seed_count": True,
            "same_closed_loop_protocol": True,
            "same_data_split": True,
            "same_metric_schema": True,
            "same_optimizer_protocol": True,
            "same_variant_set": True,
            "unique_seeds": True,
        },
        "sources": [
            {
                "seed": source["seed"],
                "report": source["report"],
                "report_sha256": source["report_sha256"],
            }
            for source in sources
        ],
        "variants": variants,
        "limitations": [
            IFP_PROTOCOL_LIMITATION,
            "the study does not verify that optimization budget is adequate",
            "source report hashes do not re-run nested checkpoints or MuJoCo",
            "closed-loop results remain local simulation diagnostics",
            "semantic task mappings are not independently verified",
            "the aggregate does not select a winner or authorize real output",
        ],
    }
    if preregistration is not None:
        assert preregistration_sha256 is not None
        study["preregistration"] = {
            "plan_sha256": preregistration_sha256,
            "hypothesis_id": preregistration.hypothesis_id,
            "claim_scope": IFP_PREREGISTRATION_CLAIM_SCOPE,
            "plan_matched": True,
            "external_preregistration_timing_verified": False,
        }
        study["endpoint_evaluation"] = evaluate_ifp_endpoints(
            preregistration,
            variants,
        )
        limitations = study["limitations"]
        assert isinstance(limitations, list)
        limitations.append("external preregistration timing is not verified")
        limitations.append(
            "preregistration does not bind the full closed-loop protocol digest"
        )
    _write_report(output, study)
    return study


def _load_preregistration(
    path: str | Path | None,
    *,
    output: Path,
) -> tuple[IFPPreregistration | None, str | None]:
    if path is None:
        return None, None
    target = Path(path).resolve()
    if target == output:
        raise IFPStudyError("IFP preregistration and study report must differ")
    try:
        source = target.read_bytes()
        plan = load_ifp_preregistration_bytes(source)
    except (OSError, IFPPreregistrationError) as error:
        raise IFPStudyError(f"invalid IFP preregistration: {error}") from error
    return plan, sha256(source).hexdigest()


def _load_source(path: Path, *, root: Path, output: Path) -> dict[str, Any]:
    target = path.resolve()
    if target == output:
        raise IFPStudyError("source report and study report must differ")
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise IFPStudyError(
            "IFP source reports must stay inside the study bundle"
        ) from error
    try:
        encoded = target.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IFPStudyError(f"failed to read IFP source report: {error}") from error
    if not isinstance(payload, dict):
        raise IFPStudyError("IFP source report must be a JSON object")
    if payload.get("schema_version") != IFP_ABLATION_SCHEMA:
        raise IFPStudyError("IFP source report has the wrong schema")
    if (
        payload.get("result") != "complete"
        or payload.get("evidence_level")
        != "offline_and_simulation_diagnostic"
        or payload.get("robot_used") is not False
        or payload.get("scientific_claim") != "not_evaluated"
    ):
        raise IFPStudyError("IFP source report has unsafe result semantics")
    comparability = _mapping(payload, "comparability")
    if any(comparability.get(key) is not True for key in _REQUIRED_COMPARABILITY):
        raise IFPStudyError("IFP source report is not internally comparable")

    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list):
        raise IFPStudyError("IFP source variants must be a list")
    variants: dict[int, Mapping[str, Any]] = {}
    for value in raw_variants:
        if not isinstance(value, Mapping):
            raise IFPStudyError("IFP source variant must be an object")
        ifp_steps = value.get("ifp_steps")
        if not isinstance(ifp_steps, int) or isinstance(ifp_steps, bool):
            raise IFPStudyError("IFP source variant requires integer ifp_steps")
        if ifp_steps in variants:
            raise IFPStudyError("IFP source variant set contains duplicates")
        variants[ifp_steps] = value
    if tuple(sorted(variants)) != IFP_ABLATION_STEPS:
        raise IFPStudyError("IFP study variant set must be K=0/2/4")

    variant_seeds = set()
    for ifp_steps, variant in variants.items():
        optimization = _mapping(variant, "optimization")
        seed = optimization.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise IFPStudyError("IFP source optimization seed is invalid")
        if optimization.get("ifp_steps") != ifp_steps:
            raise IFPStudyError("IFP source optimization variant is inconsistent")
        inference = _mapping(variant, "inference")
        if (
            inference.get("checkpoint_ifp_steps") != 0
            or inference.get("ifp_module_present") is not False
        ):
            raise IFPStudyError("IFP source retained inference-time IFP state")
        _validation(variant)
        _closed_loop(variant)
        _mapping(variant, "data")
        variant_seeds.add(seed)
    if len(variant_seeds) != 1:
        raise IFPStudyError("IFP source variants must share one seed")

    return {
        "seed": next(iter(variant_seeds)),
        "report": relative.as_posix(),
        "report_sha256": sha256(encoded).hexdigest(),
        "variants": variants,
    }


def _verify_study_contract(sources: Sequence[Mapping[str, Any]]) -> None:
    data_contracts: set[str] = set()
    optimizer_contracts: set[str] = set()
    metric_schemas: set[tuple[str, ...]] = set()
    closed_loop_contracts: set[str] = set()
    for source in sources:
        variants = source["variants"]
        assert isinstance(variants, Mapping)
        for ifp_steps in IFP_ABLATION_STEPS:
            variant = variants[ifp_steps]
            assert isinstance(variant, Mapping)
            data = dict(_mapping(variant, "data"))
            data.pop("sampling_audit", None)
            data_contracts.add(_canonical(data))

            optimizer_contracts.add(_canonical(_optimizer_protocol(variant)))

            validation = _validation(variant)
            metric_schemas.add(tuple(sorted(validation)))

            closed = dict(_closed_loop(variant))
            for field in _RESULT_FIELDS:
                closed.pop(field, None)
            closed_loop_contracts.add(_canonical(closed))

    if len(data_contracts) != 1:
        raise IFPStudyError("IFP study data split/window contract differs")
    if len(optimizer_contracts) != 1:
        raise IFPStudyError("IFP study optimizer protocol differs")
    if len(metric_schemas) != 1:
        raise IFPStudyError("IFP study metric schema differs")
    if len(closed_loop_contracts) != 1:
        raise IFPStudyError("IFP study closed-loop protocol differs")


def _optimizer_protocol(variant: Mapping[str, Any]) -> dict[str, object]:
    optimization = dict(_mapping(variant, "optimization"))
    optimization.pop("seed", None)
    optimization.pop("ifp_steps", None)
    action_decoder = optimization.pop("action_decoder", ActionDecoder.LEGACY_MEAN.value)
    if action_decoder != ActionDecoder.LEGACY_MEAN.value:
        raise IFPStudyError("IFP source action_decoder must be legacy_mean")
    optimization = {
        key: value
        for key, value in optimization.items()
        if not key.startswith(("stage1_final_", "stage2_final_"))
    }
    try:
        return normalize_optimizer_protocol(optimization)
    except IFPPreregistrationError as error:
        raise IFPStudyError(
            f"IFP source optimizer protocol is invalid: {error}"
        ) from error


def _verify_preregistration(
    plan: IFPPreregistration,
    sources: Sequence[Mapping[str, Any]],
) -> None:
    seeds = tuple(int(source["seed"]) for source in sources)
    if seeds != plan.seeds:
        raise IFPStudyError("IFP preregistration seeds do not match source reports")

    for source in sources:
        variants = source["variants"]
        assert isinstance(variants, Mapping)
        for ifp_steps in IFP_ABLATION_STEPS:
            variant = variants[ifp_steps]
            assert isinstance(variant, Mapping)
            data = _mapping(variant, "data")
            if data.get("train_split_sha256") != plan.train_split_sha256:
                raise IFPStudyError("IFP preregistration train split mismatch")
            if (
                data.get("validation_split_sha256")
                != plan.validation_split_sha256
            ):
                raise IFPStudyError("IFP preregistration validation split mismatch")
            if _optimizer_protocol(variant) != plan.optimizer_protocol:
                raise IFPStudyError("IFP preregistration optimizer protocol mismatch")
            closed_loop = _closed_loop(variant)
            if closed_loop.get("scope") != plan.closed_loop_scope:
                raise IFPStudyError("IFP preregistration closed-loop scope mismatch")
            if (
                int(closed_loop["trial_count"])
                < plan.minimum_trials_per_seed_variant
            ):
                raise IFPStudyError(
                    "IFP preregistration closed-loop trial minimum is not met"
                )


def _variant_summary(
    ifp_steps: int,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[tuple[int, str, Mapping[str, float], Mapping[str, Any]]] = []
    for source in sources:
        variants = source["variants"]
        assert isinstance(variants, Mapping)
        variant = variants[ifp_steps]
        assert isinstance(variant, Mapping)
        rows.append(
            (
                int(source["seed"]),
                str(source["report_sha256"]),
                _validation(variant),
                _closed_loop(variant),
            )
        )

    metric_keys = sorted(rows[0][2])
    validation = {
        key: _stats([row[2][key] for row in rows])
        for key in metric_keys
    }
    trial_count = sum(int(row[3]["trial_count"]) for row in rows)
    scored_count = sum(int(row[3]["scored_trial_count"]) for row in rows)
    execution_failures = sum(
        int(row[3]["execution_failure_count"]) for row in rows
    )
    success_count = sum(int(row[3]["success_count"]) for row in rows)
    weighted_error = sum(
        float(row[3]["terminal_error_mean"])
        * int(row[3]["scored_trial_count"])
        for row in rows if row[3]["scored_trial_count"] > 0
    )
    failure_counts: Counter[str] = Counter()
    for row in rows:
        failure_counts.update(_mapping(row[3], "failure_counts"))
    return {
        "ifp_steps": ifp_steps,
        "seed_count": len(rows),
        "validation": validation,
        "closed_loop": {
            "scope": rows[0][3]["scope"],
            "trial_count": trial_count,
            "scored_trial_count": scored_count,
            "execution_failure_count": execution_failures,
            "success_count": success_count,
            "success_rate": success_count / trial_count,
            "terminal_error_mean": weighted_error / scored_count if scored_count else None,
            "protocol_sha256": rows[0][3]["protocol_sha256"],
            "terminal_error_basis": rows[0][3]["terminal_error_basis"],
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "per_seed": [
            {
                "seed": seed,
                "source_report_sha256": report_sha256,
                "validation": dict(metrics),
                "closed_loop": dict(closed),
            }
            for seed, report_sha256, metrics, closed in rows
        ],
    }


def _validation(variant: Mapping[str, Any]) -> dict[str, float]:
    raw = _mapping(variant, "validation")
    if not raw:
        raise IFPStudyError("IFP source validation metrics must be non-empty")
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(float(value))
        ):
            raise IFPStudyError("IFP source validation metrics must be finite numbers")
        metrics[key] = float(value)
    return metrics


def _closed_loop(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    closed = _mapping(variant, "closed_loop")
    try:
        ClosedLoopResult.from_payload(closed)
    except IFPAblationError as error:
        raise IFPStudyError(f"IFP source {error}") from error
    return closed


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "sample_std": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise IFPStudyError(f"IFP source report is missing {key}")
    return nested


def _canonical(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise IFPStudyError("IFP source contract is not JSON-safe") from error


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise IFPStudyError(f"IFP study report already exists: {path}") from error
    except OSError as error:
        raise IFPStudyError(f"failed to write IFP study report: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate comparable K=0/2/4 IFP reports across seeds."
    )
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args(argv)
    try:
        study = aggregate_ifp_study(
            args.reports,
            report_path=args.report,
            preregistration_path=args.preregistration,
        )
    except IFPStudyError as error:
        parser.error(str(error))
    print(json.dumps(study, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "IFP_STUDY_SCHEMA",
    "IFPStudyError",
    "MIN_STUDY_SEEDS",
    "aggregate_ifp_study",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
