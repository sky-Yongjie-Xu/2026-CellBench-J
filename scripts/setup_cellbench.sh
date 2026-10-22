#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
conda_cmd="${CONDA_EXE:-/root/anaconda3/bin/conda}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${XDG_CACHE_HOME:-$HOME/.cache}/cellbench/conda-pkgs}"
export PYTHONNOUSERSITE=1
cd "$project_dir"

if ! "$conda_cmd" run -n cellbench python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    "$conda_cmd" env create -f environment-cellbench.yml -y
fi
"$conda_cmd" env config vars set -n cellbench PYTHONNOUSERSITE=1

run_python() { "$conda_cmd" run --no-capture-output -n cellbench python "$@"; }
run_python -m pip install 'pip<26' 'setuptools<81' wheel numpy==1.26.4
run_python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118
if [[ -f requirements-cellbench.lock.txt ]]; then
    run_python -m pip install -r requirements-cellbench.lock.txt
else
    run_python -m pip install -r requirements-cellbench.txt
fi

# This vendored upstream version exposes the tokenizer/pretrainer APIs used by CellBench.
stage_dir="$(mktemp -d -t cellbench-geneformer.XXXXXX)"
trap 'rm -rf -- "$stage_dir"' EXIT
cp -a public_code/LangCell/geneformer_001 "$stage_dir/geneformer_001"
run_python -m pip install --no-deps "$stage_dir/geneformer_001"
run_python - <<'PY'
import importlib.util
from pathlib import Path
import shutil

package = Path(importlib.util.find_spec('geneformer').origin).parent
for source, target in (
    ('token_dictionary_gc30M.pkl', 'token_dictionary.pkl'),
    ('gene_median_dictionary_gc30M.pkl', 'gene_median_dictionary.pkl'),
):
    shutil.copy2(Path('weights/Geneformer') / source, package / target)
PY
run_python -m pip check
run_python scripts/check_cellbench_environment.py
