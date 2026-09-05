# Capacidades, compatibilidade e limites

[Início](../README.md) · [Instalação](INSTALL.md) · [Validação real](UNIVERSAL_VALIDATION_2026-09-05.md)

O Crono Matrix é uma camada de controle do llama.cpp. Ele não substitui o
motor de inferência e não altera os pesos para tornar um modelo multimodal.
Seu diferencial é conectar metadados, recursos da máquina, configuração
editável e confirmação do servidor em um fluxo compartilhado pelas duas UIs.

## Inventário de recursos

| Área | Implementação atual | Condições e limites |
| --- | --- | --- |
| Hardware | CPU/threads/NUMA, RAM/swap, GPU/VRAM e armazenamento | Caminhos atuais concentrados em Linux/NVIDIA; não é detecção universal de todos os aceleradores |
| GGUF | Metadados, arquitetura, quantização, dimensões, experts e cabeçalhos de tensores; descoberta de shards | Compatibilidade depende do leitor e do llama.cpp selecionados; não materializa pesos para apenas inspecionar metadados |
| Arquitetura | Estimativas de atenção, estados recorrentes, MoE e adaptações específicas existentes | Ter um adaptador no código não comprova qualidade em toda variante/finetune |
| Contexto | Janela nativa, estimativa de memória, Fit e contexto efetivo do runtime | Janela alocada não garante qualidade em toda a extensão |
| KV cache | Tipos K/V, offload, slots e recálculo | Suporte/compatibilidade do tipo dependem da arquitetura e backend |
| Offload | Camadas na GPU, experts na CPU e planos nativos por tensor | Maior ocupação da VRAM não é, sozinha, maior velocidade |
| Execução | Flash Attention, batch/ubatch, threads, sampling e preview do comando | Opções dependem das capacidades do binário; alterações manuais passam pela validação |
| Sampling | Metadados de geração, perfis existentes e controles manuais | Hardware não determina temperatura ideal; não existe receita única de qualidade |
| MMProj | Validação de encoders visuais/áudio e execução na CPU quando selecionada | Mesmo tamanho de projeção não prova qualidade/compatibilidade sem teste real |
| Raciocínio | Detecção pelo template/runtime, preservação e perfis de esforço/orçamento | Formato, template e cliente devem suportar o controle; budget não é garantia de qualidade |
| Especulação | Controles de MTP, n-gram e draft quando disponíveis | MTP pode consumir memória adicional; ganho de velocidade exige medição de aceitação |
| Processo | Inicialização, parada, logs, erros e verificação de saúde/runtime | Mantenha portas locais; não equivale a um serviço endurecido para internet pública |
| Memória host | Estimativa de crescimento, checkpoints/prompt cache, monitor C99 e swap NVMe | Linux; swap depende de disco e permissões e não elimina riscos de pressão/OOM |
| Hugging Face | Busca, radar de lançamentos/atualizações, seleção e download, inclusive no desktop | Rede opcional; evento no radar não comprova que o arquivo local mudou |
| Atualização local | Verificação de origem e identidade conforme dados disponíveis | Veja os estados e a comparação de hash no guia de atualização de modelos |
| Alpha Eval | Suite de 23 eixos, seed, repetições, sampling e raciocínio configuráveis | Requer Node funcional e servidor; pontuação não substitui seus workloads |
| Desktop | CustomTkinter com operação offline e recursos online opcionais | Precisa de ambiente gráfico/Tcl-Tk; não requer navegador nem servidor FastAPI em execução |
| Web | Painel local com o mesmo núcleo do desktop | Não confundir porta do launcher com porta da API/WebUI do llama.cpp |

## Perfil universal: quatro camadas distintas

1. **Modelo:** arquitetura e contexto nativo informados pelo GGUF validado.
2. **Projetor:** encoders disponíveis, separados por modalidade.
3. **Servidor:** modelo, contexto, modalidades e template realmente ativos.
4. **Harness:** formatos e funcionalidades que o cliente consegue consumir.

O arquivo `.crono-agent/agent-local.json` publica contexto, compactação sugerida,
modalidades, raciocínio, informações de ferramentas e dados do runtime.
`agent-local.env.sh` fornece variáveis de conexão. O adaptador OpenCode
traduz o perfil para seu provedor; outros clientes precisam consumir os campos
que suportam ou ser configurados manualmente.

| Informação | Comportamento do produto |
| --- | --- |
| ID do modelo | Alias reportado pelo servidor prevalece sobre nome antigo do formulário |
| Contexto efetivo | Janela confirmada por `/props`, usada no perfil e na recomendação de compactação |
| Contexto nativo | GGUF validado; zero indica desconhecido, não contexto ilimitado |
| Entradas | Texto, imagem, vídeo e áudio conforme runtime; PDF não é anunciado como entrada nativa |
| Saídas | O adaptador atual publica texto; vocoder encontrado não basta para prometer áudio na API |
| Esforço | Off/Low/Medium/High/Max quando a integração detecta raciocínio; budgets finitos são sugestões de controle ao servidor |
| Ferramentas | Suporte do template e definições do endpoint nativo são informações distintas |
| Compactação | Recomendação ao cliente; o launcher não obriga todo harness a implementar compactação |
| Limite de resposta | Compartilha a janela com prompt/histórico/ferramentas; não é reserva independente de tokens |

As modalidades publicadas incluem origem em `capabilities.evidence`:
`runtime_props`, `gguf_preview` ou `unknown`. Sem confirmação suficiente,
o produto não deve anunciar uma capacidade como comprovada.

## Ferramentas nativas não são o MCP privado

O produto preserva a integração das ferramentas do próprio llama-server e
patches para navegador Playwright, diretório de trabalho e telemetria de
raciocínio. A disponibilidade exata depende do build, das permissões e do
endpoint `/tools`. Chromium é instalação opcional.

O servidor MCP privado do Crono Matrix **não acompanha o repositório**.
Configurações de MCP externo podem ser encaminhadas ao suporte upstream,
mas isso é separado das ferramentas nativas e não instala um MCP privado.
Uma ferramenta de navegador habilitada pode acessar a rede; modo local não
equivale a isolamento de rede das ferramentas.

## Cobertura comprovada nesta rodada

- Leitura e planejamento reais de 4Beasts Q6_K (Qwen35MoE) e Nemotron Omni
  Q4_K_M (Nemotron-H-MoE), na RTX 3060 12 GiB.
- Carga do Nemotron com 262.144 tokens e MMProj completo na CPU.
- Respostas corretas aos testes curtos de texto com raciocínio, PNG e WAV.
- Vídeo anunciado pelo runtime; **não verificado** com clipe nessa rodada.
- 105 testes Python do produto; não equivalem a 105 benchmarks de modelos.

Windows/macOS, AMD/Intel/Apple GPU, outras arquiteturas, limites completos de
raciocínio, contexto longo preenchido e operação end-to-end em cada harness
exigem validação adicional. Consulte as condições no [relatório](UNIVERSAL_VALIDATION_2026-09-05.md).

## O que não prometemos

Não há garantia de ganho fixo em tokens/s, qualidade idêntica entre tipos de
KV, ausência de OOM, nem ajuste globalmente ótimo para qualquer GGUF. A
matriz de importância (imatrix) usada na quantização não é uma camada extra
de inferência. O launcher não a transforma em ganho automático de hardware.

Para ajudar a ampliar essa cobertura, envie um relato reproduzível seguindo
[CONTRIBUTING.md](../CONTRIBUTING.md). Não é necessário compartilhar seus pesos
ou dados pessoais para contribuir com uma medição útil.
