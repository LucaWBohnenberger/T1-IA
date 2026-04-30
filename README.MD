# Projeto Jogo da Velha (Tic-Tac-Toe) com Machine Learning

Este documento descreve a estrutura do projeto de previsão de resultados do Jogo da Velha utilizando algoritmos de Machine Learning. O objetivo é desenvolver um jogo interativo onde modelos treinados possam prever, a cada jogada, o desfecho da partida (vitória de X, vitória de O, empate ou jogo em andamento).

## Divisão de algoritmos (Será apagado futuramente esse tópico)
Antes de seguirmos com o readme, só para que fique estruturado quem é responsavel por cada parte, coloquem seus algoritmos que irão utilizar aqui embaixo, para não repetirmos os mesmos, se tiverem dúvidas sobre quais algoritmos usar, me avisem que posso ajudá-los a escolher algum:

- MLP
- k-NN
- Arvores de Decisão
- Tsetlin Machine - Luca W.B.
- ...   

## Estrutura do Projeto

/
├── Dados/
│   ├── tic-tac-toe.data            # Dataset bruto original (UCI)
│   ├── preprocessed.data           # Dados com as novas amostras geradas
│   ├── data_for_ml.data            # Dados numéricos prontos para treinamento
│   └── divididos/                  # Divisão do dataset para o modelo
│       ├── train.csv               # Conjunto de treinamento
│       ├── test.csv                # Conjunto de teste
│       └── valid.csv               # Conjunto de validação
├── models/                         # Salvem os modelos treinados aqui
├── pre_processamento.py            # Script que gera os dados prontos para ML
└── game.py                         # Script do jogo interativo e plotagem das previsões

## Pasta `Dados/`

Aqui fica toda a nossa base de informações. Se você for treinar um modelo, importe diretamente o arquivo `tic-tac-toe_ml_ready.csv`, pois ele já passou por todo o pipeline de limpeza e codificação (explicado abaixo).

## Pasta `models/`

Área de trabalho de vocês para modelagem. A ideia é que cada um treine seus algoritmos (Random Forest, XGBoost, Regressão Logística, Redes Neurais, etc.) e salve os arquivos exportados (ex: `.pkl`, `.joblib` ou `.h5`) nesta pasta. O `game.py` vai buscar os modelos daqui.

## Script `pre_processamento.py`

Este script roda o pipeline completo desde o dado bruto até o CSV final.


### Decisões importantes de pré-processamento:

#### O Problema do Dataset Original
O dataset clássico Tic-Tac-Toe Endgame só possui configurações de finais de jogo (quando alguém já ganhou ou o tabuleiro encheu). A classe "not finish" (jogo em andamento) era inexistente (0 casos).

#### Data Augmentation 

- **Criando "not finish"**: Peguei jogos empatados (tabuleiros cheios) e apaguei aleatoriamente de 1 a 8 posições para simular jogos no meio da partida, gerando nossa nova classe, coloquei um total de 200 casos nela.

- **Criando mais empates ("tie")**: Como o dataset só tinha 16 empates, criei um gerador que embaralha 5 peças "X" e 4 peças "O" ou vice e versa e valida se o resultado é um empate real, aumentando nossa base de dados para balancear o target.

#### Reprodutibilidade
Todas as seeds globais (Random, NumPy, OS Hash) foram travadas em 42. Se rodarem o script novamente, o dataset gerado será exatamente o mesmo.

#### One-Hot Encoding & Dummy Trap
Transformei as letras (x, o, b) do tabuleiro em dados numéricos usando One-Hot Encoding. Um detalhe vital: usei `drop_first=True`. Isso significa que a coluna do espaço vazio (b) foi descartada para evitar a Armadilha das Variáveis Dummy (multicolinearidade perfeita). O modelo deduzirá que a casa está vazia quando as colunas de "X" e "O" daquela casa forem ambas 0.

#### Classes do Target
O label encoding da resposta final ficou mapeado da seguinte forma:

| Código | Classe |
|--------|--------|
| 0 | X win |
| 1 | O win |
| 2 | tie (empate) |
| 3 | not finish (em andamento) |

## Script `game.py` (Próximos Passos)

Este script vai conter a interface/lógica principal do jogo da velha.

### O que precisamos implementar aqui juntos:

- [X] A mecânica do jogo (X e O alternando turnos)
- [] Carregamento dos algoritmos salvos na pasta `models/`
- [] A cada jogada feita pelo usuário, o estado atual do tabuleiro deve passar pelo mesmo processamento numérico e ser injetado nos modelos
- [] Plotar na tela (seja via terminal ou interface gráfica) o resultado da previsão (X win, O win, tie ou not finish) para cada modelo, mostrando a confiança de cada um.

---

**Qualquer dúvida sobre a formatação dos dados, me deem um toque ou deem uma olhada no código do `pre_processamento.py`**
