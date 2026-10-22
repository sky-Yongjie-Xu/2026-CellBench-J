#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/cellbench-pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/cellbench-uv-cache}"
runtime_root="${CELLBENCH_ENV_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/cellbench/runtime_envs}"
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
uv_cmd="${UV_EXE:-/root/.local/bin/uv}"
base_python="${CELLBENCH_PYTHON:-/root/anaconda3/envs/cellbench/bin/python}"
model="${1:-genecompass}"
prepare_environment() {
    local target="$runtime_root/$1"
    mkdir -p "$runtime_root" runtime_envs
    if [[ -d "runtime_envs/$1" && ! -L "runtime_envs/$1" ]]; then
        target="$project_dir/runtime_envs/$1"
    fi
    if [[ ! -x "$target/bin/python" ]]; then
        if [[ "${3:-}" == shared ]]; then
            "$uv_cmd" venv --python "$2" --system-site-packages "$target"
        else
            "$uv_cmd" venv --python "$2" "$target"
        fi
    fi
    if [[ ! -e "runtime_envs/$1" && ! -L "runtime_envs/$1" ]]; then
        ln -s "$target" "runtime_envs/$1"
    fi
}
case "$model" in
  genecompass)
    prepare_environment genecompass "$base_python" shared
    "$uv_cmd" pip install --python runtime_envs/genecompass/bin/python transformers==4.30.0 tokenizers==0.13.3 numpy==1.26.4
    "$uv_cmd" pip install --python runtime_envs/genecompass/bin/python --no-deps peft==0.4.0 accelerate==0.20.3
    ;;
  transcriptformer)
    "$uv_cmd" python install 3.11
    prepare_environment transcriptformer 3.11
    "$uv_cmd" pip install --python runtime_envs/transcriptformer/bin/python torch==2.5.1 --index-url "${CELLBENCH_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
    "$uv_cmd" pip install --python runtime_envs/transcriptformer/bin/python ./public_code/TranscriptFormer
    ;;
  scprint2)
    "$uv_cmd" python install 3.11
    prepare_environment scprint2 3.11
    "$uv_cmd" pip install --python runtime_envs/scprint2/bin/python torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "${CELLBENCH_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
    "$uv_cmd" pip install --python runtime_envs/scprint2/bin/python ./public_code/scPRINT2 performer-pytorch==1.1.4 local-attention==1.9.15 pyarrow==19.0.1
    ;;
  cellfm)
    prepare_environment cellfm "$base_python" shared
    "$uv_cmd" pip install --python runtime_envs/cellfm/bin/python "${CELLBENCH_MINDSPORE_SPEC:-mindspore==2.6.0}" numpy==1.26.4
    ;;
  uce|sclong)
    "$base_python" -c 'import accelerate, torch_geometric, local_attention; print("Shared cellbench dependencies available")'
    ;;
  *) echo 'Usage: bash scripts/setup_frontier_environments.sh {uce|sclong|genecompass|transcriptformer|scprint2|cellfm}' >&2; exit 2 ;;
esac
