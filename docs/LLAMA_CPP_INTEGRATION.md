# Integração com llama.cpp

O produto fixa uma revisão exata porque a integração usa APIs experimentais do
servidor e patches na WebUI/ferramentas. O fluxo reproduzível é:

1. clonar `ggml-org/llama.cpp`;
2. fazer checkout da revisão em `third_party/llama.cpp/REVISION`;
3. validar/aplicar `patches/llama.cpp/crono-matrix.patch`;
4. copiar `patches/llama.cpp/overlay/`;
5. compilar no diretório `build-crono`.

O patch local inclui:

- ferramenta nativa `browser_playwright`;
- diretório de trabalho Unicode enviado no corpo da chamada de ferramenta;
- controles e sincronização de modelo/raciocínio na WebUI;
- contagem de `reasoning_tokens` nas respostas OpenAI-compatible;
- fontes experimentais de MetaHead preservadas no checkout local;
- falha de alocação MMProj reportada sem abortar o processo inteiro;
- testes upstream ampliados para esses comportamentos.

Uma atualização upstream exige merge manual. Nunca aplique `git pull` sobre o
checkout modificado como atualização de produto. Primeiro teste o patch contra
o novo commit, resolva o delta, regenere o SHA-256 e execute regressão real.

## Seleção do build

Quando o usuário aponta para a raiz de um checkout, o launcher não considera um
build válido apenas porque `llama-server` existe. Ele executa `--version` no
servidor e no `llama-fit-params` correspondente, rejeitando árvores com
bibliotecas compartilhadas ausentes. O planejamento e a execução sempre usam o
mesmo par de binários.

## MMProj e capacidades publicadas

O MMProj é validado pelos próprios metadados GGUF. Visão, entrada de áudio e
saída de áudio são capacidades independentes; o nome do arquivo não é usado
como prova. Entre vários projetores na pasta do modelo, o launcher prioriza um
projetor compatível que reúna mais modalidades e publica no perfil universal:

- contexto nativo do GGUF e contexto efetivo confirmado por `/props`;
- entradas de texto, imagem, vídeo e áudio realmente ativas;
- presença de encoder/vocoder de geração separada da capacidade da API: o
  adaptador atual publica saída de texto, sem prometer geração de áudio apenas
  porque encontrou um arquivo de vocoder;
- ferramentas nativas e suporte do template a chamadas de ferramenta;
- raciocínio, orçamento e compactação calculados sobre o contexto efetivo.

Em máquinas nas quais pesos + contexto + MMProj não cabem juntos na VRAM, o
MMProj pode permanecer na CPU com `--no-mmproj-offload --mmproj-device none`.
Isso preserva VRAM para os pesos quentes e para KV/estado recorrente; não
desativa imagem, vídeo ou áudio de entrada.

Na WebUI nativa, uma mídia anexada diretamente à mensagem é enviada como
`image_url`, `input_video` ou `input_audio`. Nesse primeiro turno, o patch
oculta somente a ferramenta `read_media`, que exige um caminho local e pode
induzir modelos multimodais a ignorar o conteúdo já anexado. A ferramenta
volta a ficar disponível nos turnos seguintes e continua funcionando para
arquivos indicados por caminho; as demais ferramentas permanecem ativas.

## Contrato de planejamento e publicação

O contexto nativo vem do GGUF validado. `native_context_window = 0` significa
desconhecido, não uma janela de 4K inferida dos valores iniciais do leitor.
O contexto utilizável e o ponto sugerido de compactação seguem o `n_ctx`
efetivo de `/props`, mesmo quando o formulário ainda contém outro valor.
Ampliar ou reduzir a janela em execução não altera o contexto nativo publicado.

As modalidades têm origem explícita em `capabilities.evidence`:

- `runtime_props`: modalidades informadas pelo servidor carregado;
- `gguf_preview`: previsão baseada nos encoders do projetor válido e ativado,
  antes de haver uma resposta de runtime;
- `unknown`: sem confirmação suficiente. Um runtime antigo sem esses campos
  não é anunciado como multimodal apenas pela existência do MMProj.

Um projetor só de áudio não implica visão. Vídeo exige confirmação do runtime,
pois um encoder visual não comprova sozinho o suporte do caminho de vídeo.
O harness também precisa implementar os formatos anunciados; o perfil não
adiciona capacidades ausentes no modelo, no backend ou no cliente.

O cache do plano `llama-fit-params` inclui a identidade de hardware já detectada,
RAM/VRAM disponíveis e dispositivos visíveis no ambiente, além de GGUF, binário
e parâmetros. Uma nova leitura com disponibilidade diferente exige novo plano.
Falha do executável ou plano inválido não pode entrar nesse cache. A chamada
nativa continua sendo uma estimativa de alocação, não um benchmark de geração.

Sampling não deve depender da quantidade de VRAM. Metadados de geração são a
base genérica; receitas específicas existentes e edições explícitas do usuário
são preservadas. Arquiteturas ainda não validadas e hardware fora da cobertura
atual precisam de testes: não há garantia de ótimo global de velocidade,
contexto e qualidade para qualquer GGUF. Em especial, a detecção de hardware
atual ainda contém caminhos Linux/NVIDIA que exigem adaptação para outros sistemas.

Regressão do contrato: `tests/test_universal_contract.py` usa metadados e saída
de fit sintéticos. Seus resultados não são evidência de qualidade nem de
desempenho de inferência.
