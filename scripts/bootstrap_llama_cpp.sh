#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision="$(tr -d '[:space:]' < "$project_root/third_party/llama.cpp/REVISION")"
repository="${CRONO_LLAMA_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"
llama_dir="${CRONO_LLAMA_CPP_DIR:-$project_root/llama.cpp}"
build_dir="${CRONO_LLAMA_BUILD_DIR:-$llama_dir/build-crono}"
patch_file="$project_root/patches/llama.cpp/crono-matrix.patch"
overlay_dir="$project_root/patches/llama.cpp/overlay"
expected_patch_sha="4193d2f585f5c007742bbd799e39a10e47a57042123c1ada3e625f745b64ee94"
with_browser=0
prepare_only=0
force_cpu=0

usage() {
  cat <<'EOF'
Uso: ./scripts/bootstrap_llama_cpp.sh [opções]

  --with-browser  instala Chromium para browser_playwright
  --prepare-only  clona, fixa e aplica os patches sem instalar/compilar
  --cpu           força build somente CPU
  -h, --help      mostra esta ajuda
EOF
}

while (($#)); do
  case "$1" in
    --with-browser) with_browser=1 ;;
    --prepare-only) prepare_only=1 ;;
    --cpu) force_cpu=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opção desconhecida: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for tool in git cmake sha256sum; do
  command -v "$tool" >/dev/null || {
    echo "Dependência ausente: $tool" >&2
    exit 1
  }
done

actual_patch_sha="$(sha256sum "$patch_file" | awk '{print $1}')"
if [[ "$actual_patch_sha" != "$expected_patch_sha" ]]; then
  echo "Patch local foi alterado sem atualizar sua revisão/hash." >&2
  echo "Esperado: $expected_patch_sha" >&2
  echo "Atual:    $actual_patch_sha" >&2
  exit 1
fi

if [[ ! -e "$llama_dir/.git" ]]; then
  if [[ -e "$llama_dir" ]]; then
    echo "Destino existe, mas não é um checkout Git: $llama_dir" >&2
    exit 1
  fi
  git clone --filter=blob:none "$repository" "$llama_dir"
fi

current_revision="$(git -C "$llama_dir" rev-parse HEAD 2>/dev/null || true)"
marker="$llama_dir/.crono-patch-applied"
marker_value=""
if [[ -f "$marker" ]]; then
  marker_value="$(tr -d '[:space:]' < "$marker")"
fi

if [[ "$current_revision" == "$revision" && "$marker_value" == "$expected_patch_sha" ]]; then
  echo "Checkout e patch Crono Matrix já estão preparados."
else
  if [[ -n "$(git -C "$llama_dir" status --porcelain 2>/dev/null)" ]]; then
    echo "O checkout llama.cpp contém alterações locais não reconhecidas." >&2
    echo "Use um destino limpo ou preserve essas mudanças manualmente: $llama_dir" >&2
    exit 1
  fi
  if ! git -C "$llama_dir" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$llama_dir" fetch --depth=1 origin "$revision"
  fi
  git -C "$llama_dir" checkout --detach "$revision"
  git -C "$llama_dir" apply --check "$patch_file"
  git -C "$llama_dir" apply "$patch_file"
  cp -a "$overlay_dir/." "$llama_dir/"
  printf '%s\n' "$expected_patch_sha" > "$marker"
fi

if ((prepare_only)); then
  echo "llama.cpp preparado em $llama_dir"
  exit 0
fi

for tool in npm; do
  command -v "$tool" >/dev/null || {
    echo "Dependência ausente: $tool (necessária para a WebUI modificada)" >&2
    exit 1
  }
done

npm ci --prefix "$llama_dir/tools/ui"
if ((with_browser)); then
  npm exec --prefix "$llama_dir/tools/ui" playwright install chromium
fi

cmake_args=(
  -S "$llama_dir"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=ON
  -DGGML_LTO=ON
  -DGGML_CPU_REPACK=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_TESTS=ON
  -DLLAMA_BUILD_UI=ON
  -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_CURL=ON
)

if ((force_cpu == 0)) && command -v nvidia-smi >/dev/null && command -v nvcc >/dev/null; then
  cmake_args+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES=native
    -DGGML_CUDA_FA_ALL_QUANTS=ON
    -DGGML_CUDA_GRAPHS=ON
    -DGGML_CUDA_COMPRESSION_MODE=speed
  )
  echo "Backend selecionado: CUDA para a GPU instalada."
else
  cmake_args+=(-DGGML_CUDA=OFF)
  echo "Backend selecionado: CPU nativa."
fi

cmake "${cmake_args[@]}"
jobs="${CRONO_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
cmake --build "$build_dir" --config Release -j "$jobs" \
  --target llama-server llama-fit-params

server="$build_dir/bin/llama-server"
[[ -x "$server" ]] || {
  echo "Build terminou sem produzir $server" >&2
  exit 1
}

echo "llama.cpp pronto: $server"
