# Verificação de atualização dos GGUF locais

## O que a interface compara

O radar do Hugging Face acompanha repositórios e lançamentos. Ele não prova
que uma quantização local mudou. A lista **Modelos locais** agora possui o
botão `⇄ Verificar origem e atualizações dos GGUF`, que consulta o arquivo
exato correspondente ao GGUF instalado.

Para cada arquivo, a verificação usa, nesta ordem:

1. a origem registrada no arquivo `<modelo>.gguf.crono-origin.json`;
2. para modelos antigos sem manifesto, uma associação automática pelo nome
   exato do arquivo, somente quando o Hugging Face retorna uma única origem;
3. o tamanho remoto e, quando disponível, o SHA-256 do blob LFS.

O hash local é calculado em segundo plano. Portanto, um GGUF grande não trava
a interface nem interrompe o servidor em execução. O resultado aparece no
item local:

| Estado | Significado |
| --- | --- |
| `ATUAL` | tamanho e SHA-256 local conferem com o arquivo remoto |
| `ATUALIZAÇÃO DISPONÍVEL` | tamanho/hash remoto mudou e o arquivo local ainda corresponde ao download registrado |
| `ARQUIVO DIVERGENTE` | o arquivo local foi alterado ou não corresponde à origem registrada |
| `SEM ORIGEM` | nenhum repositório foi encontrado pelo nome exato |
| `FONTES AMBÍGUAS` | mais de um repositório possui o mesmo nome de arquivo |
| `NÃO VERIFICADO` | o servidor não forneceu hash ou a consulta remota falhou |

## Manifesto e downloads novos

Downloads feitos pela interface já gravam no manifesto o repositório, revisão,
commit, nome do shard, tamanho, SHA-256 e data da transferência. Modelos
multipartes recebem um manifesto por shard. O manifesto é auxiliar: ele não é
carregado como modelo e não altera os parâmetros de inferência.

Modelos antigos não são marcados como atuais apenas porque o nome parece
correto. A primeira verificação precisa consultar a origem. Uma associação
única é persistida para que as próximas consultas não dependam de busca por
texto.

## Limitações intencionais

- Um repositório atualizado não implica que todos os seus GGUFs foram
  regenerados; por isso a comparação é feita por arquivo.
- Sem SHA-256 remoto, o mesmo tamanho não é prova de identidade e permanece
  `NÃO VERIFICADO`.
- Uma fonte ambígua não é escolhida silenciosamente.
- O botão faz uma consulta explícita; o radar periódico não calcula hashes
  automaticamente.

## Reproduzir por teste

```bash
python -m unittest tests.test_launcher_runtime -v
```

Os testes cobrem confirmação de hash, persistência do manifesto, atualização
remota, divergência local e nomes sem origem ou ambíguos.
