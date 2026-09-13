#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_dir"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${SO101_WAM_PYTHON:-python}
benchmark_dir=$(mktemp -d)

cleanup() {
  rm -rf -- "$benchmark_dir"
}

trap cleanup EXIT

"$python_bin" -c 'import mujoco; print("MuJoCo", mujoco.__version__)'
"$python_bin" -m pytest -q \
  tests/test_mujoco.py \
  tests/test_mujoco_benchmark.py \
  tests/test_mujoco_semantic_benchmark.py \
  tests/test_mujoco_semantic_prompt_controls.py \
  tests/test_mujoco_semantic_suite.py \
  tests/test_mujoco_prompt_controls.py \
  tests/test_prompt_directionality.py \
  tests/test_semantic_prompt_directionality.py
"$python_bin" -m so101_wam.mujoco_cli \
  --config configs/mujoco.toml \
  --steps 2
"$python_bin" -m so101_wam.mujoco_benchmark \
  --config configs/mujoco.toml \
  --manifest configs/mujoco_heldout_benchmark.json \
  --artifact-dir "$benchmark_dir/artifacts" \
  --report "$benchmark_dir/report.json"
