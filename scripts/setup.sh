#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
cc_bin="${CC:-cc}"

command -v "$python_bin" >/dev/null || {
  echo "Python 3 não encontrado: $python_bin" >&2
  exit 1
}
command -v "$cc_bin" >/dev/null || {
  echo "Compilador C não encontrado: $cc_bin" >&2
  exit 1
}

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$project_root/.venv"
fi

"$project_root/.venv/bin/python" -m pip install --upgrade pip
"$project_root/.venv/bin/python" -m pip install \
  -r "$project_root/requirements-web.txt" \
  -r "$project_root/requirements-gui.txt"

mkdir -p "$project_root/.crono-native" "$project_root/.crono-runtime/uploads"
"$cc_bin" -O3 -std=c99 -Wall -Wextra -pthread \
  "$project_root/native/crono_memory_guard.c" \
  -o "$project_root/.crono-native/crono-memory-guard"

echo "Ambiente Python e monitor C99 preparados."
echo "Próximo passo: ./scripts/bootstrap_llama_cpp.sh"
