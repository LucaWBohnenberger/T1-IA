import sys
import os
import warnings
import joblib
import random
from collections import Counter
import pathlib
import numpy as np
import pickle

from sklearn.exceptions import InconsistentVersionWarning

import Models.DecisionTree.DecisionTree as DecisionTree
import Models.MLP.MLP as MLP
from Models.TsetlinMachine.TsetlinMachine import TsetlinMachine

# --- MAPEAMENTO DE MÓDULOS ---
sys.modules["DecisionTree"] = DecisionTree
sys.modules["MLP"] = MLP

# --- MAPEAMENTO DE CLASSES ---
TARGET_CLASSES = {
    0: "Vitória do X",
    1: "Vitória do O",
    2: "Empate",
    3: "Jogo em andamento",
}

# Pesos na votação do ensemble (demais modelos = 1.0).
ENSEMBLE_VOTE_WEIGHT = {
    "DecisionTree": 1.0,
    "Knn": 1.0,
    "MLP": 1.0,
    "SVM": 1.0,
    "TsetlinMachine": 1.0,
}


def _ensemble_vote_weight(model_name):
    return ENSEMBLE_VOTE_WEIGHT.get(model_name, 1.0)


def load_models(models_dir="Models"):
    base = pathlib.Path(models_dir)
    DecisionTree_path = base / "DecisionTree" / "decision_tree.joblib"
    Knn_path = base / "Knn" / "best_knn_model.joblib"
    Mlp_path = base / "MLP" / "model.pkl"
    Svm_path = base / "SVM" / "best_svm_model.joblib"
    TM_path = base / "TsetlinMachine" / "model.npz"

    models = []

    # Artefatos foram serializados com outra versão do sklearn; carregar é seguro na prática,
    # mas o sklearn emite InconsistentVersionWarning em cada unpickle.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)

        # 1. DecisionTree
        if DecisionTree_path.exists():
            try:
                models.append(("DecisionTree", joblib.load(DecisionTree_path)))
            except Exception as e:
                print(f"Erro ao carregar DecisionTree: {e}")

        # 2. KNN
        if Knn_path.exists():
            try:
                models.append(("Knn", joblib.load(Knn_path)))
            except Exception as e:
                print(f"Erro ao carregar Knn: {e}")

        # 3. MLP
        if Mlp_path.exists():
            try:
                with open(Mlp_path, "rb") as f:
                    carregado = pickle.load(f)
                if isinstance(carregado, dict):
                    mlp_model = next(
                        (v for k, v in carregado.items() if hasattr(v, "predict")), None
                    )
                    if mlp_model is None:
                        raise ValueError("Nenhum objeto com 'predict' encontrado.")
                else:
                    mlp_model = carregado
                models.append(("MLP", mlp_model))
            except Exception as e:
                print(f"Erro ao carregar MLP: {e}")

        # 4. SVM
        if Svm_path.exists():
            try:
                models.append(("SVM", joblib.load(Svm_path)))
            except Exception as e:
                print(f"Erro ao carregar SVM: {e}")

        # 5. Tsetlin Machine
        if TM_path.exists():
            try:
                model = TsetlinMachine.load(str(TM_path))
                models.append(("TsetlinMachine", model))
            except Exception as e:
                print(f"Erro ao carregar TsetlinMachine: {e}")

    return models


def encode_board(board):
    features = []
    for row in board:
        for cell in row:
            if cell == "o":
                features.extend([1, 0])
            elif cell == "x":
                features.extend([0, 1])
            else:
                features.extend([0, 0])
    return [features]


def update_metrics(metrics, model_name, pred, true_class):
    """Atualiza o dicionário de pontuação."""
    if model_name not in metrics:
        metrics[model_name] = {"acertos": 0, "erros": 0}

    if pred == true_class:
        metrics[model_name]["acertos"] += 1
    else:
        metrics[model_name]["erros"] += 1


def predict_ensemble(models, board, metrics, true_class):
    """Faz a predição, avalia contra o resultado real e retorna o texto formatado."""
    if not models:
        return "Nenhum modelo disponível para previsão."

    X_input = np.array(encode_board(board))
    predictions = []

    for name, model in models:
        try:
            pred = int(model.predict(X_input)[0])
            predictions.append((name, pred))
        except Exception as e:
            print(f"Erro na previsão do modelo {name}: {e}")

    if not predictions:
        return "Falha nas previsões."

    # Votação ponderada (SVM e TsetlinMachine pesam mais; MLP 1.5×; restante 1.0×).
    pesos_por_classe = Counter()
    for name, pred in predictions:
        pesos_por_classe[pred] += _ensemble_vote_weight(name)
    vencedor_voto = max(
        pesos_por_classe.items(),
        key=lambda item: (item[1], -item[0]),
    )[0]

    # Atualização das métricas
    update_metrics(metrics, "Ensemble", vencedor_voto, true_class)
    for name, p in predictions:
        update_metrics(metrics, name, p, true_class)

    # Formatação das strings para impressão
    detalhes = " | ".join(
        [f"{name[:4]}...: {TARGET_CLASSES.get(p, str(p))}" for name, p in predictions]
    )
    resultado_final = TARGET_CLASSES.get(vencedor_voto, "Desconhecido")
    resultado_real = TARGET_CLASSES.get(true_class, "Desconhecido")

    # Montando a saída visual
    saida = f"\n[STATUS REAL DO TABULEIRO] => {resultado_real}\n"
    saida += f"[PREVISÃO DO ENSEMBLE]   => {resultado_final}\n"
    saida += f"[Votos Individuais] {detalhes}\n\n"

    saida += "--- PLACAR DE DESEMPENHO DA IA ---\n"
    # Printa o Ensemble primeiro, depois os modelos individuais
    ordem_print = ["Ensemble"] + [m[0] for m in models]
    for name in ordem_print:
        if name in metrics:
            acertos = metrics[name]["acertos"]
            erros = metrics[name]["erros"]
            total = acertos + erros
            acc = (acertos / total) * 100 if total > 0 else 0
            saida += f" {name:<15}: Acertos: {acertos:<2} | Erros: {erros:<2} | Acurácia: {acc:05.1f}%\n"

    return saida


# --- LÓGICA DO JOGO ---


def start_game():
    return [
        ["b", "b", "b"],
        ["b", "b", "b"],
        ["b", "b", "b"],
    ]


def print_board(board):
    print()
    for row in board:
        print(" " + " | ".join(cell.upper() if cell != "b" else " " for cell in row))
        print("---+---+---")


def get_empty_positions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == "b"]


def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "b":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "b":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != "b":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "b":
        return board[0][2]
    return None


def is_draw(board):
    return all(cell != "b" for row in board for cell in row)


def get_true_class(board):
    """Mapeia o estado real do tabuleiro para a classe correspondente."""
    winner = check_winner(board)
    if winner == "x":
        return 0
    elif winner == "o":
        return 1
    elif is_draw(board):
        return 2
    else:
        return 3


def player_move(board):
    while True:
        try:
            move = input("\nDigite sua jogada (linha e coluna, ex: 0 2): ")
            i, j = map(int, move.split())
            if i not in range(3) or j not in range(3):
                print("Posição fora do limite! Use números de 0 a 2.")
                continue
            if board[i][j] != "b":
                print("Posição já ocupada!")
                continue
            board[i][j] = "x"
            break
        except ValueError:
            print("Entrada inválida! Digite dois números separados por espaço.")


def random_ai_move(board):
    empty = get_empty_positions(board)
    if empty:
        i, j = random.choice(empty)
        board[i][j] = "o"
        print(f"\nIA jogou na posição: {i} {j}")


def play():
    models = load_models()
    metrics = {}  # Dicionário que vai guardar os acertos/erros durante a partida
    board = start_game()

    print("\n" + "=" * 40)
    print("Bem-vindo! Você é 'X'. A IA (Aleatória) é 'O'.")
    print("=" * 40)

    # Imprime o tabuleiro vazio inicial e já faz a primeira predição do Ensemble
    print_board(board)
    true_class = get_true_class(board)
    print(predict_ensemble(models, board, metrics, true_class))

    while True:
        # ==========================================
        # 1. TURNO DO JOGADOR (X)
        # ==========================================
        player_move(board)
        print_board(board)

        # Avalia o tabuleiro  após a sua jogada
        true_class = get_true_class(board)
        print(predict_ensemble(models, board, metrics, true_class))

        # Verifica se a sua jogada encerrou o jogo
        if true_class == 0:
            print("\n🎉 FIM DE JOGO! Você venceu o jogo!")
            break
        elif true_class == 2:
            print("\n🤝 FIM DE JOGO! Deu velha! (Empate)")
            break

        # ==========================================
        # 2. TURNO DA MÁQUINA (O)
        # ==========================================
        random_ai_move(board)
        print_board(board)

        # Avalia o tabuleiro após a jogada da IA
        true_class = get_true_class(board)
        print(predict_ensemble(models, board, metrics, true_class))

        if true_class == 1:
            print("\n💀 FIM DE JOGO! A IA venceu o jogo!")
            break
        elif true_class == 2:
            print("\n🤝 FIM DE JOGO! Deu velha! (Empate)")
            break


if __name__ == "__main__":
    play()
