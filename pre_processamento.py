import pandas as pd
from pathlib import Path
import numpy as np
import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


path_data = Path("Dados") / "tic-tac-toe.data"
data = pd.read_csv(path_data, sep=",", header=None)

matrix = data.to_numpy()


def preprocessing(game):
    """
    Função para ajustar as classes para que elas batam com o que foi pedido (X win, O win, not finish, tie)
    """
    # Checar Linhas e Colunas
    for i in range(3):
        # Verificando Linhas
        if (
            game[0 + (3 * i)] == "x"
            and game[1 + (3 * i)] == "x"
            and game[2 + (3 * i)] == "x"
        ):
            game[9] = "X win"
            return game
        if (
            game[0 + (3 * i)] == "o"
            and game[1 + (3 * i)] == "o"
            and game[2 + (3 * i)] == "o"
        ):
            game[9] = "O win"
            return game

        # Verificando Colunas
        if game[i] == "x" and game[i + 3] == "x" and game[i + 6] == "x":
            game[9] = "X win"
            return game
        if game[i] == "o" and game[i + 3] == "o" and game[i + 6] == "o":
            game[9] = "O win"
            return game

    # Checar Diagonais
    if (game[0] == "x" and game[4] == "x" and game[8] == "x") or (
        game[2] == "x" and game[4] == "x" and game[6] == "x"
    ):
        game[9] = "X win"
        return game

    if (game[0] == "o" and game[4] == "o" and game[8] == "o") or (
        game[2] == "o" and game[4] == "o" and game[6] == "o"
    ):
        game[9] = "O win"
        return game

    # Checar se o jogo não terminou
    espacos_vazios = ["b", " ", ""]
    if any(espaco in game[:9] for espaco in espacos_vazios):
        game[9] = "not finish"
        return game

    # Caso de Empate
    game[9] = "tie"
    return game


matrix_processada = np.apply_along_axis(preprocessing, axis=1, arr=matrix)
contagem_pandas = pd.Series(matrix_processada[:, 9]).value_counts()
print(contagem_pandas)


def create_not_finish(matrix, num):
    """
    Create new games "not finish" with the "tie" games

    Args:
        matrix (np.ndarray): Matrix with games already labeled correctly
        num (int): Number of new games "not finish" to generate
    """
    games_tie = matrix[matrix[:, 9] == "tie"]

    new_games = []

    for _ in range(num):
        # Sorteia um jogo de empate aleatório (com reposição)
        game = random.choice(games_tie).copy()
        qtd_apagar = random.randint(1, 8)
        positions = random.sample(range(9), qtd_apagar)

        for i in positions:
            game[i] = "b"

        game[9] = "not finish"
        new_games.append(game)

    return np.array(new_games)


def create_ties(num):
    """
    Gera tabuleiros cheios aleatórios e filtra apenas os que resultam em empate.
    """
    new_ties = []

    # Um tabuleiro cheio sempre tem 5 'X' e 4 'O', ou vice versa
    base1 = ["x", "x", "x", "x", "x", "o", "o", "o", "o"]
    base2 = ["x", "x", "x", "x", "o", "o", "o", "o", "o"]

    while len(new_ties) < num:
        # Embaralha as 9 peças para criar um cenário de tabuleiro aleatório
        base = random.choice((base1, base2))
        tabuleiro = random.sample(base, 9)

        # Adiciona um espaço no final da lista para representar a coluna 9
        jogo_teste = tabuleiro + [""]

        jogo_avaliado = preprocessing(jogo_teste)
        if jogo_avaliado[9] == "tie":
            new_ties.append(jogo_avaliado)

    return np.array(new_ties)


ties = create_ties(184)
matrix_more_ties = np.vstack((matrix_processada, ties))
not_finish = create_not_finish(matrix_more_ties, 200)
final_matrix = np.vstack((matrix_more_ties, not_finish))


# Embaralhar o data_set e salvar
np.random.shuffle(final_matrix)
df_final = pd.DataFrame(final_matrix)
path = Path("Dados") / "preprocessed.data"
df_final.to_csv(path, index=False, header=False)

df_ml = df_final.copy()
X = df_ml.iloc[:, :-1]
y = df_ml.iloc[:, -1]

# one hot encoding
# Os dados serão representados em duas colunas para cada posição, um significa X e outra O, se ambas estiverem vazias, representa espaço vazio
X_encoded = pd.get_dummies(X, columns=X.columns, drop_first=True)


# Transformando em classes numericas
map_classes = {"X win": 0, "O win": 1, "tie": 2, "not finish": 3}
y = y.map(map_classes)


df_for_ml = pd.concat([X_encoded, y], axis=1)
path = Path("Dados") / "data_for_ml.data"
df_for_ml.to_csv(path, index=False, header=None)
