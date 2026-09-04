# Política de segurança

## Superfície de ataque

O Crono Matrix pode iniciar `llama-server`, executar ferramentas nativas no
host, ler modelos e criar/remover um arquivo de swap administrado. Por padrão,
as interfaces e a API devem permanecer em `127.0.0.1`.

Não publique uma instância com ferramentas de escrita/shell em uma interface
de rede não confiável. Use autenticação, sandbox/container e uma lista explícita
de origens antes de qualquer exposição remota.

## Dados que não devem entrar em commits

GGUF/MMProj, chaves de API, arquivos `.env`, certificados, configurações do
usuário, históricos, memórias, relatórios privados e builds locais são
ignorados e verificados por `scripts/release_check.py`.

## Relato responsável

Relate vulnerabilidades de forma privada por
[GitHub Security Advisories](https://github.com/aluiziolinux/crono-matrix/security/advisories/new).
Não publique detalhes exploráveis em uma issue antes da correção. Inclua versão
ou commit, plataforma, impacto observado e uma reprodução mínima sem segredos.
