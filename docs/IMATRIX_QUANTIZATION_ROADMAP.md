# Roadmap: imatrix e quantização

Status: pendência registrada em 2026-09-01.

## Decisão conceitual

No `llama.cpp`, imatrix (importance matrix) não é uma camada de inferência
nem uma camada que deve entrar no cálculo de VRAM/KV. Ela é um arquivo de
calibração utilizado pelo `llama-quantize` para reduzir a perda de qualidade
durante a quantização.

Portanto, a imatrix não deve ser adicionada ao cálculo de:

- VRAM do modelo carregado;
- KV cache;
- contexto;
- batch ou ubatch;
- número de camadas offload;
- RAM residente do servidor.

Ela deve participar do fluxo de quantização, da identificação da origem da
quantização e da avaliação de qualidade do artefato gerado.

## Estado observado

- O launcher já exclui arquivos com `imatrix` e `importance-matrix` da lista de
  modelos carregáveis como se fossem modelos principais.
- O leitor GGUF identifica a quantização pelo nome do arquivo, mas ainda não
  expõe um estado formal `imatrix_present`, `imatrix_source` ou
  `quantization_calibration`.
- Os perfis de inferência tratam `Q6_K`, `Q4_K`, `IQ*` e outros tipos como
  quantizações de runtime, mas não distinguem se o arquivo foi calibrado com
  imatrix.
- A UI não mostra se um GGUF possui evidência de calibração imatrix.
- Não deve ser inferido que todo arquivo `IQ*` tenha uma imatrix associada.

## Implementação planejada

### 1. Detecção segura

Adicionar ao metadata do modelo:

```text
imatrix_present: bool
imatrix_path: string | null
imatrix_format: gguf | dat | unknown
imatrix_entries: integer | null
imatrix_dataset: string | null
imatrix_compatible: bool | unknown
quantization_source: native | requantized | unknown
```

A descoberta deve procurar apenas arquivos candidatos no mesmo diretório ou
em uma pasta de calibração explicitamente configurada. O nome do arquivo, por
si só, não será tratado como prova suficiente.

### 2. Compatibilidade

Antes de associar uma imatrix a um modelo, verificar:

- arquitetura e versão do modelo-base;
- tensor names presentes;
- dimensão de cada entrada da imatrix;
- número de experts quando o modelo for MoE;
- tipo de quantização de destino;
- dataset e quantidade de chunks, quando disponíveis;
- hash do modelo-base ou metadado equivalente.

Se a compatibilidade não puder ser comprovada, mostrar `DESCONHECIDA` e não
usar a imatrix automaticamente.

### 3. Quantização

O fluxo de quantização deverá aceitar explicitamente:

```text
--imatrix /caminho/imatrix.gguf
--include-weights ...
--exclude-weights ...
--tensor-type ...
```

A aplicação deve registrar no artefato e no relatório:

- comando completo;
- commit do `llama.cpp`;
- hash do modelo de entrada;
- hash da imatrix;
- dataset;
- tipo final de cada tensor relevante;
- tamanho final;
- perplexity/KLD, quando o teste estiver disponível.

### 4. Autotuning de inferência

O autotuning não deve alterar KV, contexto ou offload por causa da imatrix.
Deve, contudo, usar a informação de quantização para:

- distinguir `Q*` de `IQ*` corretamente;
- evitar assumir que dois Q6_K possuem a mesma qualidade;
- escolher kernels compatíveis com o tipo real do tensor;
- informar que a qualidade depende da calibração usada;
- separar desempenho de runtime da qualidade de quantização.

### 5. Avaliação

Modelos quantizados com e sem imatrix devem ser comparados com o mesmo:

- prompt;
- seed;
- sampling;
- contexto;
- número de tokens;
- build do `llama.cpp`.

O relatório deve separar:

```text
qualidade do modelo-base
perda da quantização
efeito da imatrix
velocidade de inferência
uso de memória
```

## Critérios para o GitHub

Antes da publicação, o repositório deverá conter:

- este roadmap;
- um manifesto do build do `llama.cpp`;
- hash do modelo e da imatrix usada em cada quantização;
- comandos reproduzíveis;
- distinção entre arquivos de modelo, calibração e runtime;
- testes que confirmem que `.imatrix.gguf` não é iniciado como modelo;
- teste de associação compatível e incompatível;
- documentação sobre licença e origem do dataset de calibração.

## Prioridade

1. P1: expor o estado de imatrix sem alterar silenciosamente o autotuning.
2. P1: impedir associação automática quando a compatibilidade for incerta.
3. P2: integrar o fluxo de quantização com comando e relatório reproduzíveis.
4. P2: adicionar avaliação A/B de qualidade e velocidade.
5. P3: apresentar selo visual na UI: `IMATRIX CONFIRMADA`, `AUSENTE` ou
   `DESCONHECIDA`.

## Regra de segurança

Nunca afirmar que um modelo é “imatrix” apenas porque o nome contém `IQ`,
`Q6_K`, `Q4_K` ou `imatrix`. A confirmação deverá vir dos metadados, do
arquivo de calibração compatível ou de documentação verificável do artefato.

## Objetivo maior: autotuning universal

O objetivo do Crono Matrix é possuir um planejador universal capaz de analisar
qualquer GGUF e o hardware disponível, configurando o servidor para extrair o
máximo desempenho possível sem sacrificar qualidade ou estabilidade.

O planejador deverá separar quatro dimensões:

1. **Modelo** — arquitetura, camadas, MoE, SSM/Mamba, atenção, contexto nativo,
   MTP, multimodalidade, quantização e estado da imatrix.
2. **Hardware** — GPU, VRAM livre, CUDA/driver, CPU, threads, RAM disponível,
   NUMA, armazenamento e pressão de memória.
3. **Restrições** — contexto solicitado, qualidade mínima, visão, ferramentas,
   limite de RAM, slots, MTP e política de swap.
4. **Medição** — prompt processing, geração, uso de VRAM/RAM, falhas de
   alocação, temperatura, estabilidade e qualidade observada.

O resultado não deverá vir de um perfil fixo por nome de modelo. Perfis
específicos podem servir como ponto inicial, mas o planejador deverá validar,
medir e ajustar cada decisão no hardware real.

### Fase 0 obrigatória: perfilamento do ambiente

Antes de selecionar ou calcular parâmetros para qualquer modelo, o launcher
deverá analisar o ambiente do usuário. Nenhum perfil anterior poderá ser
aplicado como se fosse universal.

O perfilamento deverá coletar, quando disponível:

- sistema operacional, kernel e arquitetura;
- backend realmente disponível;
- GPU, VRAM total e VRAM livre;
- versão do driver e capacidade CUDA/Vulkan/Metal;
- CPU, arquitetura SIMD, núcleos físicos e lógicos;
- frequência e topologia NUMA;
- RAM total, disponível, pressão e swap;
- armazenamento, espaço livre e suporte a NVMe;
- versão, commit, hash e flags do `llama-server` selecionado;
- flags efetivamente aceitas pelo binário;
- presença e capacidade de `llama-fit-params`;
- permissões para cgroup, systemd scope e monitoramento.

O resultado será um `HardwareProfile` com timestamp, validade e nível de
confiança por campo. Leituras ausentes ou suspeitas serão marcadas como
`unknown`, nunca substituídas silenciosamente por valores da máquina de
desenvolvimento.

Esse perfil deverá ser feito por um caminho rápido, com cache invalidado quando
mudar o driver, o build, o dispositivo ou a disponibilidade de memória. O
benchmark de inferência ficará para a fase de calibração posterior.

O mesmo perfil deverá identificar a plataforma de execução:

```text
EnvironmentProfile
├── os: linux | windows | macos | other
├── distro_or_build
├── kernel_or_version
├── architecture: x86_64 | arm64 | ...
├── shell
├── package_manager
├── compiler_and_version
├── cmake_or_ninja
├── git
├── gpu_toolchain
└── permission_capabilities
```

A detecção usará as APIs nativas do sistema e verificações reais dos
executáveis, não apenas o nome informado pelo usuário. Em Linux, verificará
kernel, distribuição, compilador, CMake/Ninja, CUDA e permissões de usuário.
Em Windows, verificará versão do Windows, PowerShell, Git, CMake e MSVC/Visual
Studio ou WSL. Em macOS, verificará macOS, Xcode/clang e Metal. A presença de
uma ferramenta só será considerada válida se ela responder ao comando de
versão ou a um teste equivalente.

O instalador/compilador será escolhido por uma matriz de plataforma:

| Plataforma | Caminho preferencial |
| --- | --- |
| Linux | build nativo com CMake/Ninja e backend detectado |
| Windows | MSVC/CMake; WSL somente como opção explícita |
| macOS | clang/CMake com Metal quando disponível |
| outra | runtime pré-compilado ou instalação manual |

Se faltarem ferramentas, o launcher deverá listar exatamente o que falta e
pedir permissão antes de instalar dependências do sistema. Nenhum gerenciador de
pacotes, PowerShell, WSL ou script remoto será executado silenciosamente.

Depois da compilação, deverá validar o executável produzido com `--version`,
`--help`, backend e hash do binário. O resultado será gravado no
`RuntimeProfile`, junto com a plataforma e o commit utilizado.

### Fase 1 obrigatória: validar o `llama.cpp`

Depois do perfilamento do hardware e antes de calcular os parâmetros do modelo,
o launcher deverá validar o runtime selecionado:

- raiz do repositório;
- presença de `.git` ou instalação identificável;
- branch e commit completo;
- estado sujo ou modificações locais;
- binário `llama-server` correspondente;
- executáveis auxiliares, incluindo `llama-fit-params` e `llama-quantize`;
- versão retornada por `--version`;
- flags retornadas por `--help`;
- backend compilado e backend realmente inicializável;
- compatibilidade entre fonte, build e hardware detectado.

Se o runtime estiver ausente, incompleto ou incompatível, a interface deverá
informar o motivo e oferecer opções explícitas:

```text
usar outro llama.cpp já instalado
instalar do repositório oficial
selecionar manualmente um runtime
cancelar
```

A instalação automática do repositório oficial só poderá ocorrer após
autorização explícita do usuário. Antes da confirmação, o launcher deve mostrar
o diretório destino, URL, branch/commit alvo, comandos de build, espaço
necessário e alterações previstas.

Não será permitido executar `pull`, `reset`, `checkout`, rebase ou substituição
de binário sobre uma árvore com modificações locais sem uma confirmação
específica. O caminho seguro é preservar a instalação atual e criar uma cópia
ou build paralelo; a troca para o novo runtime deve ser atômica e reversível.

O resultado deverá gerar um `RuntimeProfile` contendo o hash do commit, hash do
binário, flags disponíveis, backends e estado de validação. O autotuning só
deve usar uma configuração como medida quando esse perfil estiver confirmado.

### Ordem universal de decisão

```text
descobrir GGUF
→ identificar arquitetura e recursos
→ descobrir hardware real
→ estimar memória por componente
→ gerar candidatos de configuração
→ eliminar candidatos inviáveis
→ medir candidatos seguros
→ escolher fronteira desempenho × qualidade
→ iniciar servidor
→ validar /props e telemetria reais
→ aprender somente com medições persistidas
```

Nenhuma otimização poderá declarar sucesso apenas por estimativa. O sistema
deve diferenciar `estimado`, `medido`, `confirmado` e `não verificado`.

## Artefato confirmado para teste

O diretório `modelos/4BeastsOfApocalypse.Q6_K/` contém:

```text
4BeastsOfApocalypse.Q6_K.gguf
4BeastsOfApocalypse.imatrix.gguf
```

O segundo arquivo foi inspecionado com o `gguf_dump` do próprio `llama.cpp` e
possui:

```text
general.type = imatrix
imatrix.datasets = imatrix-training-full-3
imatrix.chunk_count = 319
imatrix.chunk_size = 512
1020 entradas/tensores de calibração
```

O modelo principal declara `QWEN35MOE`, 256 experts, 8 experts usados, 40
camadas e `Q6_K`. Como a imatrix foi publicada no mesmo pacote pelo autor da
quantização, o launcher deve tratar os dois arquivos como um único artefato
`IMATRIX_PAIRED` para fins de identificação, documentação e testes.

Não será exigida uma confirmação adicional de origem para carregar ou executar
o modelo. A imatrix não será carregada pelo `llama-server` durante a inferência
e não alterará o cálculo de VRAM, KV, contexto, batch ou offload.

### Proveniência confirmada pelo artefato

O próprio GGUF contém:

```text
general.url = https://huggingface.co/mradermacher/4BeastsOfApocalypse-GGUF
mradermacher.quantized_by = mradermacher
mradermacher.quantize_version = 2
```

Assim, a imatrix local deve ser tratada como a imatrix publicada junto da
quantização `mradermacher` do repositório indicado, e não como uma imatrix
genérica inventada pelo launcher. Essa informação serve apenas para exibição,
reprodutibilidade e classificação do artefato; não deve criar uma etapa de
confirmação inútil no fluxo normal.
