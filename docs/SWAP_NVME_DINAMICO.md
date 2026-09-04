# Swap NVMe dinâmico do Crono Matrix

## Objetivo

O swap NVMe é uma proteção contra encerramento por falta de memória ao abrir
GGUFs maiores que a RAM física. Ele não é tratado como RAM rápida e não entra
no cálculo de contexto/KV para aumentar artificialmente a janela.

A ordem de uso esperada nesta máquina é:

1. RAM física;
2. pesos GGUF mapeados diretamente do NVMe com `mmap`, quando necessário;
3. swapfile NVMe do Crono Matrix, prioridade calculada acima da ZRAM;
4. ZRAM existente como última proteção comprimida.

Na máquina de referência, a ZRAM usa prioridade `100` e o Crono Matrix usa
`110`. A prioridade antiga `10` estava errada: ela fazia o kernel comprimir o
modelo dentro da própria RAM até a máquina entrar em thrashing, enquanto o
swap NVMe permanecia vazio.

## Monitor de RAM em C99

Cada `llama-server` iniciado pelo launcher recebe um monitor nativo compilado
de `native/crono_memory_guard.c`. O binário usa C99/POSIX, é compilado com
`-O3 -march=native -flto` e acompanha `MemAvailable` diretamente em
`/proc/meminfo`, sem depender do intervalo de atualização da interface.

Não existe reserva rígida de RAM. Em Linux com cgroup v2 e systemd de usuário,
o launcher inicia o `llama-server` em um scope exclusivo, sempre com
`memory.high=max`. O monitor apenas publica uma advertência quando
`MemAvailable` fica abaixo de 1.536 MiB; ele não altera o teto do processo.

Essa decisão foi tomada por medição. Com um 35B MoE residente, o controle
adaptativo de `memory.high` gerou 1.158 eventos de throttling, 61% de PSI full,
evicção/refault de pesos quentes e queda de geração de cerca de 22,5 para
17,6 tokens/s. Restaurar `memory.high=max` reduziu imediatamente a pressão.
Chamadas explícitas a `memory.reclaim` também foram descartadas por poderem
bloquear dentro do kernel.

A proteção efetiva contra OOM é a hierarquia nativa do Linux: RAM, descarte de
páginas GGUF mapeadas, swap NVMe prioritário e somente depois ZRAM. O kernel
possui informação de recência e pressão que o launcher não
consegue reproduzir com um teto de usuário. O C99 permanece útil para telemetria
de baixa latência sem interferir na inferência.

O monitor não encerra o modelo e não reduz contexto, KV ou offload. O kernel
escolhe as páginas frias do cgroup; isso preserva a carga, mas uma carga que
acuse uso sustentado do NVMe pode sofrer latência por paginação. Não existe
mais loop de monitoramento Python durante a inferência. Se o C99 não puder ser
compilado, a carga é recusada; uma falha posterior do monitor fica visível nos
logs e na telemetria.

## Cálculo por modelo

Depois de ler os metadados e tensores do GGUF, `OptimalParams` estima:

- pesos/expert tensors colocados no host;
- MMProj em CPU, quando a visão está ativa nessa configuração;
- KV em CPU, quando `--no-kv-offload` estiver ativo;
- capacidade do KV ativo no dispositivo que o atende (VRAM quando
  `--kv-offload` está ativo); essa memória não deve ser forçada para swap;
- limite de `--cache-ram`, que é o cache de prompts ociosos em RAM;
- cópias de estado de `--ctx-checkpoints`, buffers independentes no host que
  não obedecem ao limite de `--cache-ram`;
- workspace e estado fixo do runtime;
- reserva adaptativa de 5% da RAM, limitada entre 1 e 2 GiB;
- limite de checkpoints calculado pela capacidade RAM + swap NVMe, evitando
  multiplicar cegamente 32 checkpoints por snapshots de KV de vários GiB.

Quando a carga residente não cabe integralmente na RAM, `none` só é aceito se
o conjunto realmente ativo couber na memória física e o excedente frio couber
em um swap NVMe com prioridade superior à ZRAM. Sem essa rota comprovada, o
launcher seleciona `mmap` ou recusa uma escolha manual insegura. A recomendação
de swap considera a maior lacuna entre o pico no host e o tamanho total do
modelo, adiciona 4 GiB de emergência e arredonda em blocos de 4 GiB. O resultado
mínimo é 8 GiB e o máximo é 64 GiB, limitado pelo espaço que deixe pelo menos
32 GiB livres no NVMe.

O arquivo de swap é pré-alocado no disco antes de iniciar o processo, mas o
Linux não oferece uma API para reservar páginas de swap para um único buffer
ou obrigar o KV ativo a residir nele. Isto é intencional: trocar KV a cada
token faria a geração perder desempenho de forma severa. A reserva evita falta
de capacidade quando checkpoints/cache crescem; o kernel continua escolhendo
somente páginas frias para ZRAM/NVMe.

O swap pode crescer automaticamente antes da inicialização do servidor. Ele
não é reduzido automaticamente: encolher swap ativo exigiria descarregar
páginas e poderia interromper uma carga estável.

## Arquivos e persistência

- swapfile: `~/.local/share/crono-matrix/swapfile`
- gerenciador: `scripts/manage_nvme_swap.sh`
- persistência: entrada exclusiva em `/etc/fstab`
- prioridade: `max(prioridade da ZRAM) + 10` (normalmente `110`)
- backup inicial: `/etc/fstab.crono-matrix.bak`

Em Btrfs, o arquivo é criado com `btrfs filesystem mkswapfile`; não é um
arquivo esparso comum. A criação, crescimento e remoção exigem autenticação
administrativa via `pkexec`.

## Controles

Na telemetria web e na página **Sistema** da aplicação desktop são exibidos:

- ZRAM total e usada;
- swap NVMe total e usado;
- tamanho recomendado para o modelo selecionado;
- motivo do cálculo;
- estado do crescimento automático.

Também existem controles para aplicar o tamanho calculado, ativar/desativar o
crescimento automático e remover somente o swapfile pertencente ao Crono
Matrix.

## Operação manual e diagnóstico

```bash
scripts/manage_nvme_swap.sh status
pkexec scripts/manage_nvme_swap.sh create 16
pkexec scripts/manage_nvme_swap.sh remove
swapon --show
```

O tamanho aceito pelo script fica entre 8 e 64 GiB. A remoção atua somente no
caminho fixo do Crono Matrix e na linha correspondente do `fstab`.

## Limites de desempenho

Swap evita OOM; ele não torna um modelo maior rápido. Se `USED` do swap NVMe
crescer durante geração sustentada, a carga está sob pressão real de memória.
Nesse caso, para recuperar desempenho deve-se reduzir contexto, slots,
batch/ubatch, visão ou tensors residentes — e não aumentar ainda mais o swap.
