# Crono Matrix

### Seu modelo. Seu hardware. Você no controle.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-42f58d.svg)](LICENSE)
[![CI](https://github.com/aluiziolinux/crono-matrix/actions/workflows/ci.yml/badge.svg)](https://github.com/aluiziolinux/crono-matrix/actions/workflows/ci.yml)

**Execute modelos GGUF grandes com planejamento orientado ao hardware,
desktop offline e painel Web local — usando o llama.cpp de verdade.**

Escolher um modelo é fácil. Decidir quanto contexto cabe, onde colocar os
experts de um MoE e como dividir memória entre pesos, KV cache e visão é a
parte difícil. O Crono Matrix reúne essas decisões em um painel: lê o GGUF,
detecta recursos disponíveis, propõe uma configuração e permite inspecionar
o comando antes de executar. Depois do carregamento, consulta o servidor
para confirmar o que realmente ficou ativo.

[Começar agora](docs/INSTALL.md) · [Capacidades e limites](docs/CAPABILITIES.md) ·
[Resultados reais](docs/UNIVERSAL_VALIDATION_2026-09-05.md) · [English](README.en.md)

> **Alpha, com testes em hardware real.** A cobertura atual se concentra em
> Linux/NVIDIA. O objetivo é ampliar a compatibilidade, não prometer o máximo
> de desempenho para qualquer modelo ou computador sem medir.

## Por que experimentar?

- **Modelos grandes, decisões explicáveis.** Contexto, GPU offload, MoE na CPU,
  KV e memória de trabalho participam do planejamento. Você vê o comando
  gerado e pode ajustar os parâmetros.
- **Duas interfaces, um único núcleo.** Desktop CustomTkinter sem navegador ou
  painel Web local: ambos usam o mesmo cálculo e gestão de processos.
- **Offline quando você quiser.** Com dependências, modelo e build instalados,
  a inferência é local. Radar Hugging Face, downloads e ferramentas de rede
  são recursos opcionais.
- **Omni sem confundir capacidades.** Imagem e áudio são identificados
  separadamente nos projetores. O MMProj pode ficar na CPU para preservar
  VRAM. O runtime confirma as modalidades ativas.
- **Seu agente recebe informações úteis.** O modo universal publica modelo,
  contexto efetivo, modalidades e raciocínio, com adaptador para OpenCode e
  dados de conexão para outros clientes OpenAI-compatible.
- **Descubra, compare e teste.** Radar de lançamentos, downloads, verificação
  de atualizações e Alpha Eval ajudam a transformar curiosidade em
  experimentos reproduzíveis.

## O que já medimos

Em uma **RTX 3060 de 12 GiB com 31,2 GiB de RAM**, usando o build local descrito
no [relatório de validação](docs/UNIVERSAL_VALIDATION_2026-09-05.md):

| Ensaio | Resultado observado |
| --- | --- |
| Leitura dos GGUFs 4Beasts Q6_K e Nemotron Omni Q4_K_M | 0,409 s e 0,268 s; pico de aproximadamente 59 MiB de RAM no processo leitor |
| Plano do 4Beasts Q6_K | 262.144 tokens, GPU layers 40 e MoE CPU 33; inferência não repetida nesta rodada |
| Nemotron Omni carregado | 262.144 tokens por slot, KV de 816 MiB na GPU e MMProj na CPU |
| Raciocínio, imagem e áudio no Nemotron | Cálculo correto, formas/cores identificadas e áudio sintético transcrito corretamente |
| Regressão do produto em 05/09/2026 | 105 testes automatizados sem falhas |

São resultados de uma configuração específica. Contexto **alocado** não
comprova qualidade com a janela inteira preenchida. Amostras curtas não são
benchmark de código complexo nem promessa de velocidade no seu hardware.
Vídeo foi anunciado pelo runtime, mas não testado com clipe nessa rodada.

## Instale e faça seu primeiro teste

No Linux, prepare Git, Python 3.11+ com `venv` e Tcl/Tk, compiladores C/C++,
CMake, ferramentas de build e Node.js/npm. Para NVIDIA, prepare driver e CUDA
Toolkit compatíveis. Consulte o [guia completo](docs/INSTALL.md).

```bash
git clone https://github.com/aluiziolinux/crono-matrix.git
cd crono-matrix
./scripts/setup.sh
./scripts/bootstrap_llama_cpp.sh
```

Desktop:

```bash
./.venv/bin/python launch_model_gui.py
```

Ou painel Web:

```bash
./.venv/bin/python launch_model_web.py
```

Abra **http://127.0.0.1:7860**.

1. Escolha a pasta do llama.cpp e a pasta dos seus modelos.
2. Selecione o GGUF e examine o perfil proposto.
3. Para multimodalidade, forneça o MMProj correspondente e ative a opção.
4. Confira o preview do comando e inicie o servidor.
5. Teste na WebUI do llama.cpp ou ative o modo universal para seu harness.

**Nenhum modelo é incluído.** Use pesos de uma fonte confiável e respeite a
licença de cada modelo. Quer reutilizar um build existente ou compilar só
para CPU? O [guia cobre os dois caminhos](docs/INSTALL.md).

## Escolha como trabalhar

| Recurso | Para que serve |
| --- | --- |
| Desktop CustomTkinter | Operação local sem servidor Web do launcher nem navegador; recursos online opcionais |
| Painel Web | Configuração, telemetria, processo, downloads e avaliação pelo navegador local |
| Modo universal | Perfil de capacidades confirmado pelo runtime e integração com agentes |
| Ferramentas nativas | Recursos do próprio llama-server, incluindo o patch browser_playwright; independentes do MCP privado |
| Radar Hugging Face | Descoberta de modelos, eventos de atualização e consulta/download de arquivos |
| Alpha Eval | Suite de 23 eixos com seed, repetições, sampling e controles de raciocínio |
| Memória e swap NVMe | Planejamento de pressão de memória, monitor C99 e administração explícita de swap no Linux |

O harness precisa implementar os formatos anunciados. Um perfil não transforma
um modelo de texto em multimodal, nem acrescenta ferramentas ausentes no
cliente. Consulte a [matriz completa](docs/CAPABILITIES.md).

## Controle local também exige cuidado

Mantenha launcher e API em loopback. Ferramentas de arquivos, shell e navegador
podem realizar ações reais; use um diretório de trabalho dedicado e permissões
adequadas. Swap pode ajudar com falta de memória, mas NVMe não substitui a
largura de banda da RAM. Não há garantia de ausência de OOM ou travamentos.

O produto **não inclui pesos, MMProj, builds, histórico, credenciais ou o
servidor MCP privado do Crono Matrix**. O bootstrap usa uma revisão fixada do
upstream com patches rastreáveis. Não atualiza cegamente um checkout alterado.

## Ajude a ampliar a cobertura

Seu teste pode revelar o próximo ajuste importante. Abra uma
[issue](https://github.com/aluiziolinux/crono-matrix/issues) com hardware,
modelo/quantização, versão do build, comando sem segredos e resultado observado.
Relatos com medições ajudam mais que uma nota isolada de tokens/s.

Se o projeto foi útil, considere uma estrela no GitHub ou compartilhe um teste
reproduzível. Contribuições são bem-vindas: [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentação

| Documento | O que você encontra |
| --- | --- |
| [Instalação e primeiro uso](docs/INSTALL.md) | Dependências, build novo/existente, offline, atualização e solução de problemas |
| [Capacidades e compatibilidade](docs/CAPABILITIES.md) | O que é automático, confirmado e ainda exige validação |
| [Novidades](CHANGELOG.md) | Melhorias e correções por etapa |
| [Validação real](docs/UNIVERSAL_VALIDATION_2026-09-05.md) | Condições dos testes e limites das conclusões |
| [Arquitetura](docs/ARCHITECTURE.md) | Fluxo das duas interfaces até o servidor |
| [Integração llama.cpp](docs/LLAMA_CPP_INTEGRATION.md) | Revisão, patches e contrato do perfil universal |
| [Swap NVMe](docs/SWAP_NVME_DINAMICO.md) | Política de memória e monitor C99 |
| [Atualização de modelos](docs/MODEL_UPDATE_VERIFICATION.md) | Origem, hash e estados verificáveis |

## Apoie o desenvolvimento independente

O Crono Matrix é desenvolvido de forma independente. Se ele ajudar no seu
trabalho, você pode [apoiar o desenvolvimento via PIX](docs/SUPPORT.md).
O QR também aparece discretamente nas duas interfaces. A contribuição é
opcional: **nenhuma funcionalidade depende de doação**.

## Licença

[Apache License 2.0](LICENSE) para o código do Crono Matrix. Dependências e
modelos mantêm suas próprias licenças; veja [NOTICE.md](NOTICE.md).
