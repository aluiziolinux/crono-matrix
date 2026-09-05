# Novidades

As entradas descrevem mudanças do produto, não a garantia de que cada recurso
do upstream foi validado. Esta distribuição permanece **alpha**.

## 2026-09-05 — Omni, perfil universal e documentação

### Melhorias

- Detecção independente de visão, entrada de áudio e componentes de geração
  nos metadados de MMProj; preferência por projetores com mais modalidades
  entre candidatos aceitos pela verificação existente.
- Perfil universal separa contexto nativo do GGUF, contexto efetivo e
  capacidades reportadas pelo servidor, com origem explícita dos dados.
- Cache do planejamento nativo considera hardware, RAM/VRAM disponíveis e
  dispositivos visíveis; falhas/planos inválidos não entram no cache.
- Verificação executável do par llama-server/llama-fit-params antes de uso,
  com diagnóstico para bibliotecas ausentes.
- Tratamento de logs com UTF-8 e substituição de bytes inválidos.
- Documentação de instalação, capacidades, limitações, testes e versão de
  apresentação em inglês; modelo de issue para relatos reproduzíveis.
- Verificação de distribuição também inspeciona arquivos de patch e padrões
  adicionais de tokens GitHub; fixtures atuais usam caminhos genéricos.

### Correções

- Projetor de áudio não é mais confundido com visão no perfil de preview.
- Contexto solicitado antigo não prevalece sobre `/props` na publicação.
- Contexto estendido em runtime não altera o limite nativo publicado.
- Orçamento de raciocínio zero não vira ilimitado.
- Detecção de raciocínio não usa apenas `reasoning_format=none` como negativa.
- Patch da WebUI omite `read_media` no primeiro turno quando a mídia já está
  anexada, evitando a indução observada a pedir novamente um caminho de arquivo.

### Evidência

105 testes automatizados do produto sem falhas e smoke test real do Nemotron
Omni em texto, imagem e áudio. Consulte o
[relatório de 05/09](docs/UNIVERSAL_VALIDATION_2026-09-05.md). Não houve atualização
do upstream nesta etapa; a revisão fixada continua em
[`REVISION`](third_party/llama.cpp/REVISION).

### Para atualizar

Leia [Instalação — atualização](docs/INSTALL.md#atualização-preservando-seu-ambiente).
Os patches do llama.cpp mudaram em relação à versão anterior publicada: não
os reaplique cegamente sobre um checkout modificado. Use um destino separado
ou faça merge manual preservando alterações locais. Reinicie o launcher para
carregar as mudanças Python; pesos não precisam ser baixados novamente.

## Etapas anteriores no histórico

- Download de modelos Hugging Face pela interface desktop.
- Descoberta do gguf-py a partir do checkout selecionado.
- Licença Apache 2.0 e apoio voluntário via PIX nas duas interfaces.
- Empacotamento do produto sem modelos, builds ou MCP privado.
