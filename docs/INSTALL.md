# Instalação e primeiro uso

[Voltar ao início](../README.md) · [Capacidades](CAPABILITIES.md)

O caminho principal desta alpha é **Linux**, com testes reais em NVIDIA.
Windows, macOS e outros aceleradores ainda não têm validação equivalente do
produto. O suporte do llama.cpp a uma plataforma não significa que todos os
recursos do launcher já estejam portados.

## 1. Prepare o sistema

Instale pelo gerenciador de pacotes da sua distribuição:

| Dependência | Uso |
| --- | --- |
| Git | Obter o produto e a revisão do llama.cpp |
| Python 3.11+ e suporte a venv/pip | Ambiente isolado do launcher |
| Tcl/Tk para esse Python | Desktop CustomTkinter; pode ser um pacote separado |
| Compiladores C/C++, CMake e Make ou Ninja | Build do llama.cpp e monitor C99 |
| Node.js/npm compatíveis com a WebUI da revisão fixada | Build da interface nativa; Node também executa Alpha Eval |
| Driver NVIDIA e CUDA Toolkit, se usar CUDA | Backend GPU; o toolkit fornece nvcc |
| Bibliotecas/headers exigidos pelo CMake | Resolva as dependências de sistema indicadas pelo build selecionado |
| FFmpeg, quando exigido pelo caminho de mídia | Decodificação de contêineres; não necessário para iniciar o launcher |

Não existe quantidade universal mínima de RAM/VRAM: depende dos pesos,
arquitetura, contexto, KV, batch e MMProj. Reserve disco para checkout, build
e modelos. Compilar também consome RAM; evite inferência concorrente no
primeiro build.

Verificações rápidas:

```bash
python3 --version
python3 -c 'import venv, tkinter; print("venv e Tcl/Tk disponíveis")'
git --version
cmake --version
cc --version
c++ --version
node --version
npm --version
```

Para NVIDIA:

```bash
nvidia-smi
nvcc --version
```

Ter o driver funcionando não implica ter o CUDA Toolkit instalado. Não execute
o launcher como root. Consulte a
[documentação oficial de build do llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
para requisitos do backend; o produto usa a revisão fixada em
[REVISION](../third_party/llama.cpp/REVISION).

## 2. Baixe o produto

```bash
git clone https://github.com/aluiziolinux/crono-matrix.git
cd crono-matrix
./scripts/setup.sh
```

O script cria `.venv/`, instala dependências Python das duas interfaces e
compila o monitor C99. Não baixa modelos nem instala pacotes do sistema com sudo.
Para escolher outro Python instalado:

```bash
PYTHON_BIN=python3.12 ./scripts/setup.sh
```

O desktop usa [CustomTkinter 6.0.0](https://pypi.org/project/customtkinter/6.0.0/).
[Tkinter/Tcl-Tk](https://docs.python.org/3/library/tkinter.html) pertence ao
ambiente Python/sistema: instalar só CustomTkinter não substitui essa dependência.

## 3. Escolha o caminho do llama.cpp

### Criar um build do produto

```bash
./scripts/bootstrap_llama_cpp.sh
```

O bootstrap obtém o upstream oficial, fixa a revisão, verifica o SHA-256 do
patch, aplica patches/overlay e compila `llama-server` e `llama-fit-params`
em `llama.cpp/build-crono/bin/`. A WebUI nativa também é construída.

Com `nvidia-smi` e `nvcc` disponíveis, seleciona CUDA; sem eles, segue o
caminho CPU. Confira a mensagem **Backend selecionado**: um build CPU pode
ser inadequado para a velocidade esperada com um modelo grande.

O padrão paraleliza a compilação pelos processadores online. Se faltar RAM,
reduza o paralelismo, por exemplo:

```bash
CRONO_BUILD_JOBS=4 ./scripts/bootstrap_llama_cpp.sh
```

Opções adicionais:

```bash
# Build explicitamente CPU
./scripts/bootstrap_llama_cpp.sh --cpu

# Instala também o Chromium para browser_playwright
./scripts/bootstrap_llama_cpp.sh --with-browser

# Só prepara o código/patcheamento, sem npm/build
./scripts/bootstrap_llama_cpp.sh --prepare-only
```

Chromium é opcional e sua instalação exige rede. Bibliotecas adicionais de
sistema podem ser necessárias; siga o diagnóstico do instalador. Build com
arquitetura `native` é destinado à máquina onde foi compilado, não um
binário universal para distribuir a terceiros.

### Usar seu checkout/build existente

Aponte a interface para o checkout completo do llama.cpp, um diretório de
build ou o executável. O produto procura o par `llama-server` +
`llama-fit-params` e verifica se ambos executam `--version`.

O leitor `gguf-py` deve estar disponível no checkout selecionado. Recursos
dos patches do produto podem faltar em um build upstream sem patches:
[detalhes da integração](LLAMA_CPP_INTEGRATION.md).

**Não rode o bootstrap sobre um checkout modificado esperando merge
automático.** Ele recusa alterações locais não reconhecidas. Preserve seus
patches e prepare a versão do produto em outro destino:

```bash
CRONO_LLAMA_CPP_DIR="$PWD/llama.cpp-crono" ./scripts/bootstrap_llama_cpp.sh
```

## 4. Abra uma das interfaces

Desktop, sem servidor Web do launcher:

```bash
./.venv/bin/python launch_model_gui.py
```

Painel Web:

```bash
./.venv/bin/python launch_model_web.py
```

Abra **http://127.0.0.1:7860**. Essa é a porta do launcher; a API/WebUI do
llama-server usa a porta configurada no painel, normalmente 8080.

Escolha os caminhos na interface ou defina antes de iniciar:

```bash
export CRONO_MODELS_DIR="/caminho/dos/modelos"
export CRONO_LLAMA_CPP_DIR="/caminho/do/llama.cpp"
./.venv/bin/python launch_model_gui.py
```

Comece usando uma interface de cada vez para controlar a mesma instância.

## 5. Selecione e teste seu modelo

1. Adicione um GGUF à pasta escolhida; mantenha juntos todos os shards
   necessários. Pesos/projetores não acompanham o produto.
2. Aplique os caminhos, selecione o modelo e aguarde a leitura dos metadados.
3. Examine contexto, KV, offload e motivos do plano. Sampling tem relação com
   modelo/tarefa; aumentar temperatura não aumenta sua capacidade de hardware.
4. Para imagem/áudio, obtenha o MMProj correto, mantenha-o na pasta do modelo
   e ative multimodalidade. O nome do arquivo não comprova todas as modalidades.
5. Confira o comando e inicie. Aguarde o estado pronto e o contexto efetivo.
6. Faça uma pergunta curta antes de iniciar uma avaliação longa.

Para testar percepção, anexe imagem simples ou áudio conhecido. Perguntar
“você consegue ouvir?” sem anexar áudio não testa o encoder. Vídeo e formatos
dependem do modelo, projetor, build e interface.

## 6. Conecte seu harness/agente

Ative **Modo universal** no painel. Com o servidor pronto, consulte
`.crono-agent/agent-local.json`: endpoint, modelo, contexto efetivo,
capacidades e origem dessas informações. O produto tem adaptador de OpenCode;
outros clientes podem usar os dados de conexão OpenAI-compatible.

Para clientes que leem variáveis de ambiente, no mesmo terminal de execução:

```bash
source .crono-agent/agent-local.env.sh
```

Isso não configura automaticamente todo aplicativo: confira o provedor e os
campos aceitos pelo cliente. Após trocar o modelo, alguns harnesses precisam
recarregar o provedor ou abrir nova sessão. Não edite arquivos gerados durante
a execução. Se houver conflito com mudanças manuais no provedor, preserve sua
configuração; não apague o arquivo inteiro do harness.

## Offline e permissões

Com dependências, build e pesos instalados, o desktop não precisa de internet.
O painel Web também pode atender pelo loopback sem internet. Instalação,
radar, downloads e ferramentas que acessam sites precisam de rede.

Mantenha as portas em `127.0.0.1`. Habilite escrita, shell e navegador apenas
em diretórios no escopo do agente. Swap NVMe exige disco livre e operações
privilegiadas específicas: leia o [guia](SWAP_NVME_DINAMICO.md) antes de ativar.

## Atualização preservando seu ambiente

Pare avaliações e encerre o modelo pela interface. Revise mudanças locais:

```bash
git status --short
git pull --ff-only
./scripts/setup.sh
```

Se houver conflito/mudanças próprias, preserve-as; não use reset destrutivo
como solução automática. Leia o [changelog](../CHANGELOG.md). Alterações na
revisão/patch do llama.cpp podem exigir um build em **outro diretório**.
Não é preciso baixar novamente os pesos só para atualizar o launcher.

## Problemas comuns

| Sintoma | Verificação/ação |
| --- | --- |
| Repository not found | URL correta e autorização caso o repositório esteja privado |
| _tkinter ausente | Instale Tcl/Tk correspondente ao Python da venv |
| Biblioteca gguf-py não encontrada | Selecione checkout com gguf-py/gguf/__init__.py; CRONO_GGUF_PY_DIR é override avançado |
| Biblioteca compartilhada ausente no Node/llama-server | Repare dependências compatíveis; não crie symlinks entre ABIs distintas |
| CUDA não detectada/build CPU | Confira driver, nvidia-smi, nvcc e mensagem do bootstrap |
| Compilação encerrada por falta de memória | Reduza CRONO_BUILD_JOBS e libere cargas concorrentes |
| Checkout contém alterações locais | Use outro destino ou faça merge manual preservando patches |
| Porta ocupada | Encerre a instância que você iniciou ou escolha outra porta |
| Sem visão/áudio no harness | Confira MMProj ativado, /props, perfil universal e anexos no cliente |
| Contexto efetivo diferente | Compare preview, log e /props; confira memória e opções Fit/KV |
| RAM/VRAM insuficiente | Reduza contexto/batch ou reveja distribuição/quantização; swap não garante velocidade |

## Verifique / reporte um teste

```bash
make test
make release-check
```

Testes automatizados não substituem geração real. Informe sistema,
CPU/RAM/GPU, GGUF/quantização, build, comando sem segredos, sampling,
contexto efetivo e medições. Use [CONTRIBUTING.md](../CONTRIBUTING.md).
Não envie pesos, credenciais ou conversas privadas.
