import pandas as pd
from pathlib import Path
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

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

y.name = "9"
df_for_ml = pd.concat([X_encoded, y], axis=1)
path = Path("Dados") / "data_for_ml.data"
df_for_ml = df_for_ml.astype(int)
df_for_ml.to_csv(path, index=False, header=None)

print(df_for_ml.columns)
# Separando em train, val, e teste

N_PER_CLASS = 200
LABEL_COL = "9"


def _farthest_first_indices(X: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Seleciona n índices espalhados no espaço de features (Hamming em vetores binários).
    Heurística greedy tipo facility location: cada novo ponto maximiza a menor distância ao conjunto já escolhido.
    """
    m = X.shape[0]
    if n >= m:
        return np.arange(m)
    first = int(rng.randint(m))
    selected = [first]
    dist_min = np.sum(X != X[first], axis=1).astype(np.float64)
    dist_min[first] = -1.0
    while len(selected) < n:
        j = int(np.argmax(dist_min))
        selected.append(j)
        dist_min = np.minimum(dist_min, np.sum(X != X[j], axis=1).astype(np.float64))
        dist_min[selected] = -1.0
    return np.array(selected, dtype=int)


def balanced_representative_subset(
    df: pd.DataFrame,
    label_col: str,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """
    Mantém exatamente n_per_class linhas por classe, mas em geral mais representativo que sorteio puro:
    - elimina duplicatas exatas nas features (mesmo tabuleiro codificado não ocupa várias vagas);
    - se ainda houver mais linhas únicas que n_per_class, usa farthest-first para cobrir melhor o espaço de estados.
    Se houver menos linhas únicas que n_per_class, repete linhas (mesmo comportamento necessário para fechar o tamanho fixo).
    """
    rng = np.random.RandomState(seed)
    feature_cols = [c for c in df.columns if c != label_col]
    chunks = []
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls].reset_index(drop=True)
        uniq = sub.drop_duplicates(subset=feature_cols).reset_index(drop=True)
        X = uniq[feature_cols].to_numpy(dtype=np.uint8, copy=False)
        k = len(uniq)
        if k <= n_per_class:
            chunk = uniq.sample(
                n=n_per_class,
                replace=(k < n_per_class),
                random_state=seed,
            )
        else:
            idx = _farthest_first_indices(X, n_per_class, rng)
            chunk = uniq.iloc[idx].reset_index(drop=True)
        chunks.append(chunk)
    out = pd.concat(chunks, axis=0, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


df_balanced = balanced_representative_subset(df_for_ml, LABEL_COL, N_PER_CLASS, SEED)

X = df_balanced.drop(columns=["9"])
y = df_balanced["9"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.6,  # 15/25
    stratify=y_temp,
    random_state=42,
)

train_df = X_train.copy()
train_df["9"] = y_train

val_df = X_val.copy()
val_df["9"] = y_val

test_df = X_test.copy()
test_df["9"] = y_test

base_dir = Path("Dados") / "Divididos"
base_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(base_dir / "train.csv", index=False)
val_df.to_csv(base_dir / "valid.csv", index=False)
test_df.to_csv(base_dir / "test.csv", index=False)
