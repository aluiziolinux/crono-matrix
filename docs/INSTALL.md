# Instalação

## Linux

Pacotes necessários: Python 3.11 ou superior, compilador C/C++, Git, CMake e
Node.js/npm. CUDA Toolkit é necessário para compilar o backend NVIDIA.

```bash
git clone URL_DO_REPOSITORIO crono-matrix
cd crono-matrix
./scripts/setup.sh
./scripts/bootstrap_llama_cpp.sh
```

O script Python cria `.venv/`; o bootstrap cria `llama.cpp/` e
`llama.cpp/build-crono/`. Ambos são locais e ignorados pelo Git.

O leitor de metadados `gguf-py` vem do próprio checkout do `llama.cpp`; ele não
é duplicado no produto. Ao aplicar o caminho na interface, o Crono Matrix o
localiza automaticamente quando o usuário aponta para:

- o checkout `llama.cpp`;
- o diretório do projeto que contém `llama.cpp/`;
- um diretório de build;
- o executável `llama-server`.

`CRONO_GGUF_PY_DIR` permanece disponível apenas como override avançado. Se o
checkout selecionado não contiver `gguf-py/gguf/__init__.py`, execute novamente
`./scripts/bootstrap_llama_cpp.sh` ou selecione um checkout oficial completo.

Modelos não são baixados automaticamente. Guarde-os fora do repositório e
selecione sua pasta pela interface. O launcher também aceita
`CRONO_MODELS_DIR` e `CRONO_LLAMA_CPP_DIR`.

## Operação sem rede

Depois que dependências, checkout do `llama.cpp`, build e modelos estiverem no
disco, a interface CustomTkinter não necessita de FastAPI nem navegador:

```bash
./.venv/bin/python launch_model_gui.py
```

Busca/atualização no Hugging Face e instalação do Chromium são opcionais e
naturalmente exigem rede. Inferência, leitura GGUF, autotune e processo local
continuam disponíveis offline.

## Windows

O desenho do núcleo separa interfaces da execução, mas esta distribuição ainda
usa recursos Linux em proteção de memória, cgroups e swap. Windows permanece
**não verificado** e não deve ser anunciado como suportado até existir pipeline
e teste em hardware real.
