#!/usr/bin/env bash
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -n "${GCPR16_PYTHON:-}" ]]; then
  exec "$GCPR16_PYTHON" "$script_dir/run_gcpr16.py" "$@"
fi
conda_executable="${CONDA_EXE:-}"
if [[ -z "$conda_executable" ]]; then
  conda_executable="$(command -v conda || true)"
fi
if [[ -z "$conda_executable" ]]; then
  printf '%s\n' 'Conda was not found. Set GCPR16_PYTHON to the shapepriors-gcpr16 environment Python.' >&2
  exit 1
fi
exec "$conda_executable" run --no-capture-output -n shapepriors-gcpr16 python "$script_dir/run_gcpr16.py" "$@"
