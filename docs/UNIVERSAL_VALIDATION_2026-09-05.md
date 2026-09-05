# Validação do planejamento e do perfil universal — 2026-09-05

Escopo: alterações no produto, sem atualização do llama.cpp, sem alteração dos
pesos, sem escrita na configuração global dos harnesses. GPU real: RTX 3060
12 GiB; RAM total 31.2 GiB. Binário real: `build-rtx3060/bin/llama-server`,
build 10799, commit `0bcd3b97e`. Não havia `llama-server` ativo antes dos testes.

## Correções desta etapa

- Plano nativo em cache passa a considerar hardware detectado, RAM/VRAM livres
  e dispositivos visíveis. Consulta a esses valores não dispara novo processo
  de descoberta do toolkit por acerto de cache.
- Saída de um fitter com código de erro ou plano inválido não é aceita.
- Projetor de áudio não implica visão. Previsão GGUF, dados de runtime e
  capacidades desconhecidas ficam identificados no perfil.
- Contexto nativo não é inflado pela janela solicitada. Contexto de `/props`
  prevalece sobre valor antigo do formulário ou argumento de publicação.
- Budget de raciocínio igual a zero não é convertido em ilimitado.

## Leitura real e planejamento

| GGUF | Leitura | Arquitetura/layers | Plano para esta máquina |
| --- | --- | --- | --- |
| 4BeastsOfApocalypse.Q6_K | 0.409 s | QWEN35MOE; 40 layers, 10 atenção/30 recorrentes, 40 MoE | 262144 tokens, GPU layers 40, MoE CPU 33, K bf16/V q8_0 |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Q4_K_M | 0.268 s | NEMOTRON_H_MOE; 52 layers, 6 atenção/23 recorrentes/23 MoE | 262144 tokens, plano por tensor do fitter, K/V q8_0 |

CONFIRMADO: os metadados essenciais e projetores dos dois arquivos foram lidos
sem erro. Pico RSS do processo leitor: aproximadamente 59 MiB. Nenhum tensor de
peso foi materializado pelo leitor de metadados.

O cálculo levou 1.594 s e 5.795 s respectivamente, com consultas ao binário
nativo instalado. Estes números são tempos de planejamento, **não** benchmarks
de geração. A previsão de pico host era 32665 MiB no 4Beasts e 28371 MiB no
Nemotron; portanto, não equivale a uma promessa de que tudo permanecerá na RAM.
O 4Beasts não foi carregado para inferência nesta etapa.

## Inferência real do Nemotron Omni

Servidor isolado em porta efêmera de loopback, usando o comando construído pelo
produto. MMProj Q8 completo, 1079 tensores, ativado explicitamente na CPU.
Ferramentas nativas desativadas apenas neste teste de percepção, para isolar a
entrada multimodal. Pesos, contexto e KV mantiveram o plano calculado.

CONFIRMADO no log e em `/props`:

- contexto por slot: 262144; um slot;
- CUDA model buffer: 8324.53 MiB;
- CUDA_Host model buffer: 15047.42 MiB;
- CUDA KV buffer: 816.00 MiB;
- CUDA estado recorrente: 47.62 MiB;
- modalidades do runtime e perfil: texto, imagem, vídeo e áudio;
- saída publicada: texto; sem promessa de geração de áudio;
- carregamento: 26.56 s.

| Entrada sintética | Resposta observada | Tempo total |
| --- | --- | --- |
| `What is 17 multiplied by 23? Give the result.` | `391`, 37 tokens de raciocínio contabilizados | 3.11 s |
| PNG 256×256: quadrado vermelho à esquerda e círculo azul à direita | `square red, circle blue` | 7.73 s |
| WAV em inglês: “The secret code is seven four two.” | `The secret code is 742.` | 2.46 s |

Para reprodutibilidade: seed 42, temperatura 0 nas requisições de teste;
`reasoning_format=auto`. Texto com `enable_thinking=true`, budgets de 2048 e
`max_tokens=384`. Imagem/áudio com `enable_thinking=false`, `max_tokens=128`.
O WAV foi sintetizado com `espeak-ng`, voz `en-us`, velocidade 135.
Não houve alteração dos defaults de sampling do launcher.

A primeira execução do script omitiu a ativação multimodal: texto funcionou,
`/props` anunciou somente texto e a imagem foi recusada. O script foi corrigido
para ativar o MMProj e a carga foi repetida. Isso não foi contabilizado como
aprovação do teste de imagem.

As amostras curtas registraram aproximadamente 23–26 tokens/s de geração.
Não demonstram ganho de desempenho nem preservação de qualidade em código
complexo: faltam A/B e uma suite representativa. Também não demonstram que o
budget de 2048 é obedecido até o limite, pois a resposta encerrou antes dele.

Vídeo: **NÃO VERIFICADO** com clipe; capacidade apenas informada por `/props`.
Interface gráfica e cada harness: **NÃO VERIFICADOS** end-to-end nesta rodada.
Publicação/adapter: cobertos por testes automatizados de contrato.

Ao concluir, a RAM disponível era 8067 MiB; o monitor de emergência não disparou.
O processo de teste encerrou normalmente com código zero. O monitor era apenas
do ensaio, com interrupção abaixo de 768 MiB disponíveis, e não modificou o
controlador de memória do produto.

## Regressão automatizada

Comando: `.venv/bin/python -m unittest discover -s tests -q`.
Resultado: **105 testes executados sem falhas**, incluindo nove novos testes
de contrato. Metadados e respostas de fit sintéticos nesses testes não são
evidência de inferência real.

O CustomTkinter emitiu avisos de instalação de fontes bloqueada pelo sandbox;
nenhuma validação visual é deduzida dos testes Python. `release_check.py`
verificou os 82 arquivos então rastreados sem modelos/MCP. Esse ensaio ocorreu
antes do commit das alterações. Na preparação para publicação, a verificação
foi repetida sobre os 88 arquivos, incluindo nova documentação e testes, sem
modelos/MCP e com inspeção adicional dos arquivos de patch.

## Limites que permanecem

Compatibilidade universal é um objetivo, não um resultado comprovado em todas
as arquiteturas. A descoberta de hardware ainda possui caminhos Linux/NVIDIA.
São necessários adaptadores e ensaios para outros sistemas/aceleradores,
arquiteturas não testadas, contexto longo preenchido, geração complexa,
streaming e ferramentas multimodais nos harnesses. Não substituir os ajustes
medidos existentes por uma fórmula genérica sem essas evidências.
