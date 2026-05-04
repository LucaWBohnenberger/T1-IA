# Projeto Jogo da Velha (Tic-Tac-Toe) com Machine Learning

Este documento descreve a estrutura do projeto de previsão de resultados do Jogo da Velha utilizando algoritmos de Machine Learning. O objetivo é desenvolver um jogo interativo onde modelos treinados possam prever, a cada jogada, o desfecho da partida (vitória de X, vitória de O, empate ou jogo em andamento).


## Estrutura do Projeto

```text
/
├── Dados/
│   ├── tic-tac-toe.data            # Dataset bruto original (UCI)
│   ├── preprocessed.data           # Dados com as novas amostras geradas
│   ├── data_for_ml.data            # Dados numéricos prontos para treinamento
│   └── Divididos/                  # Divisão do dataset para avaliação (gerada pelo script)
│       ├── train.csv               # Conjunto de treinamento
│       ├── valid.csv               # Conjunto de validação
│       └── test.csv                # Conjunto de teste
├── Models/                         # Modelos treinados por algoritmo (joblib, pkl, npz, etc.)
├── pre_processamento.py            # Script que prepara e divide os dados para ML
└── game.py                         # Script do jogo interativo e plotagem das previsões

``` 

## Pasta `Dados/`

Aqui fica toda a base de dados. Em resumo:

- **`tic-tac-toe.data`**: arquivo bruto (UCI).
- **`preprocessed.data`**: tabuleiros já rotulados e com augmentação; ainda em formato categórico (`x`, `o`, `b`).
- **`data_for_ml.data`**: mesma base após one-hot das casas e rótulo numérico na última coluna (`9`). É o artefato “flat” antes da seleção de 800 linhas e do split.
- **`Divididos/train.csv`**, **`valid.csv`**, **`test.csv`**: conjuntos gerados pelo `pre_processamento.py` para treinar e avaliar modelos (já com cabeçalho de colunas).

Para treinar um modelo a partir do pipeline oficial, use preferencialmente os CSV em **`Dados/Divididos/`** (ou regenere-os rodando o script).

## Pasta `Models/`

Cada subpasta guarda um algoritmo e seus artefatos (por exemplo `.joblib`, `.pkl`, `.npz`). O `game.py` carrega os modelos a partir dessa estrutura.

## Script `pre_processamento.py`

Roda o pipeline ponta a ponta: leitura do UCI, rotulagem consistente, augmentação, codificação numérica, seleção de um subconjunto balanceado de tamanho fixo e gravação dos splits.

### Fluxo do pipeline

1. **Leitura** de `Dados/tic-tac-toe.data` (sem cabeçalho; 9 casas + rótulo original na coluna 10).
2. **Função `preprocessing`**: recalcula o rótulo de cada linha para uma das classes pedidas no projeto — vitória de X, vitória de O, empate (`tie`), ou jogo em andamento (`not finish`) — com base nas linhas, colunas, diagonais e presença de casas vazias.
3. **Augmentação**:
   - `create_ties(184)`: gera tabuleiros cheios aleatórios (5× e 4○ ou o contrário) e mantém só os que `preprocessing` classifica como empate.
   - `create_not_finish(..., 200)`: a partir de jogos `tie`, zera aleatoriamente entre 1 e 8 posições (`b`) para obter 200 exemplos de `not finish`.
4. **Concatenação e embaralhamento** de tudo; gravação em **`Dados/preprocessed.data`**.
5. **One-hot** das 9 posições (`pd.get_dummies`, `drop_first=True`) e **mapeamento** do alvo para inteiros; gravação em **`Dados/data_for_ml.data`**.
6. **Subconjunto balanceado representativo** (ver secção seguinte): exatamente **200 linhas por classe** (800 no total), constantes `N_PER_CLASS` e `LABEL_COL = "9"`.
7. **Split estratificado** em treino / validação / teste e gravação em **`Dados/Divididos/`**.

### Por que um subconjunto “representativo” com o mesmo tamanho?

Com quantidade fixa de amostras por classe, sortear 200 linhas ao acaso pode:

- incluir **várias duplicatas** do mesmo tabuleiro codificado, desperdiçando vagas sem aumentar diversidade;
- cobrir mal o espaço de padrões possíveis nas features binárias.

O script usa a função **`balanced_representative_subset`**:

- Em cada classe, remove **duplicatas exatas** nas colunas de entrada (mesmo vetor one-hot).
- Se ainda existirem **mais de 200 linhas distintas**, escolhe 200 com uma heurística **farthest-first** (distância de Hamming): pontos novos maximizam dispersão em relação aos já escolhidos, favorecendo **cobertura** do espaço de estados.
- Se existirem **menos de 200 únicos**, completa com **`sample(..., replace=True)`**, pois o tamanho por classe continua fixo em 200.

Assim o conjunto final mantém o mesmo **cardinal** exigido pelo trabalho, mas tende a ser **mais informativo** do que um sorteio uniforme simples.

### Divisão treino / validação / teste

Sobre o dataframe balanceado de 800 linhas:

- Primeiro `train_test_split`: **75%** treino, **25%** reserva (`test_size=0.25`), com **`stratify=y`**.
- Segundo split na reserva: **`test_size=0.6`** entre validação e teste (sobre os 25%), também estratificado.

Isso resulta em proporções da ordem de **~75% treino**, **~10% validação**, **~15% teste** do total balanceado (valores exatos dependem do arredondamento estratificado).

### Decisões complementares

#### Dataset original (UCI)

O Tic-Tac-Toe Endgame clássico traz sobretudo configurações de fim de partida; a classe “em andamento” precisa ser criada artificialmente (passo acima).

#### Reprodutibilidade

Seeds globais: `random`, NumPy e `PYTHONHASHSEED` fixadas em **42** (`SEED`), para que rerodar o script reproduza os mesmos artefatos (na mesma versão das bibliotecas).

#### One-hot e `drop_first=True`

Cada casa vira indicadores para `x` e `o`; com `drop_first=True`, evita-se redundância perfeita entre dummies. Casa vazia corresponde a **ambas** as colunas daquela posição iguais a **0**.

#### Classes do target (coluna `9`)

| Código | Classe |
|--------|--------|
| 0 | X win |
| 1 | O win |
| 2 | tie (empate) |
| 3 | not finish (em andamento) |

**Como rodar:** na raiz do projeto, `python pre_processamento.py` (gera/atualiza os arquivos em `Dados/` e `Dados/Divididos/`).

## Script `game.py` (Próximos Passos)

Este script vai conter a interface/lógica principal do jogo da velha.

### O que precisamos implementar aqui juntos:

- [X] A mecânica do jogo (X e O alternando turnos)
- [X] Carregamento dos algoritmos salvos na pasta `Models/`
- [X] A cada jogada feita pelo usuário, o estado atual do tabuleiro deve passar pelo mesmo processamento numérico e ser injetado nos modelos
- [X] Plotar na tela (seja via terminal ou interface gráfica) o resultado da previsão (X win, O win, tie ou not finish) para cada modelo, mostrando a confiança de cada um.

---

**Qualquer dúvida sobre a formatação dos dados, me deem um toque ou deem uma olhada no código do `pre_processamento.py`**
