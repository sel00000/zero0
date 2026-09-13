"""CLI for validating local compatible human/robot pair manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .paired_data import PairedDataError, load_human_robot_pairs, paired_data_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local human/robot pair manifest and print an offline "
            "integrity audit. This is not an official HumanGen importer."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        pairs = load_human_robot_pairs(args.manifest)
        audit = paired_data_audit(pairs)
    except PairedDataError as error:
        parser.error(str(error))

    print(
        json.dumps(
            dict(audit),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
