# Crono Matrix

Plano de controle local para `llama.cpp`, com cálculo automático orientado ao
hardware para modelos GGUF grandes. O mesmo núcleo atende duas interfaces:

- `launch_model_gui.py`: aplicação CustomTkinter completa; funciona sem rede
  com modelos e `llama.cpp` já presentes, e habilita recursos online somente
  quando o usuário os aciona;
- `launch_model_web.py`: interface Web local baseada em FastAPI/Jinja2.

O produto lê os metadados GGUF, observa CPU, RAM, GPU e VRAM disponíveis e
calcula contexto, KV cache, offload, camadas MoE na CPU, batch, ubatch, Flash
Attention, multimodal/MMProj e especulação quando suportada pelo binário real.
Parâmetros editados pelo usuário continuam passando pela mesma validação antes
de formar o comando efetivo do `llama-server`.

## O que deliberadamente não está no repositório

- nenhum modelo, shard GGUF, MMProj ou checkpoint;
- nenhum servidor MCP do Crono Matrix, memória, histórico ou `node_modules`;
- nenhum binário/build do `llama.cpp`;
- nenhum resultado privado de avaliação, captura de tela ou configuração local.

As ferramentas nativas do `llama-server` são independentes de MCP. Os patches
locais adicionam `browser_playwright`, propagação segura do diretório de
trabalho na WebUI e telemetria de tokens de raciocínio.

## Instalação rápida — Linux

Requisitos de sistema: Git, CMake, compilador C/C++, Python 3.11+, Node.js/npm
para a WebUI/ferramenta de navegador e, para NVIDIA, CUDA Toolkit compatível
com o driver instalado.

```bash
./scripts/setup.sh
./scripts/bootstrap_llama_cpp.sh
```

O bootstrap clona o repositório oficial na revisão fixada, valida e aplica o
patch local e compila em `llama.cpp/build-crono/`. Em máquina NVIDIA ele ativa
CUDA, arquitetura nativa, LTO, graphs e kernels de Flash Attention para todos
os tipos quantizados; sem NVIDIA, produz um build CPU nativo.

Para instalar também o Chromium usado pela ferramenta nativa:

```bash
./scripts/bootstrap_llama_cpp.sh --with-browser
```

## Execução

Desktop CustomTkinter:

```bash
./.venv/bin/python launch_model_gui.py
```

Web local:

```bash
./.venv/bin/python launch_model_web.py
```

Acesse `http://127.0.0.1:7860`. Se os GGUF estiverem fora do projeto, escolha
a pasta diretamente em qualquer uma das interfaces ou defina
`CRONO_MODELS_DIR=/caminho/dos/modelos`.

## Segurança

O launcher inicia processos, pode oferecer ferramentas nativas capazes de ler
ou alterar arquivos e pode administrar swap local mediante ação explícita.
Mantenha a API em loopback. Não exponha `llama-server` ou o launcher na rede
sem autenticação, isolamento e política de ferramentas apropriados.

## Desenvolvimento e validação

```bash
make test
make release-check
```

Veja [instalação detalhada](docs/INSTALL.md),
[arquitetura](docs/ARCHITECTURE.md) e
[integração do llama.cpp](docs/LLAMA_CPP_INTEGRATION.md).

## Estado do produto

Esta é uma versão alpha preparada para versionamento público. Linux é o alvo
confirmado nesta edição; suporte Windows ainda não foi validado. A licença
atual é source-available/proprietária. Troque-a conscientemente antes de
oferecer permissões de redistribuição ou contribuição externa.
