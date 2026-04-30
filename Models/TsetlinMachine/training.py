from TsetlinMachine import TsetlinMachine
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import optuna

# 1. Carregamento e Preparação dos Dados
data_path = Path(__file__).resolve().parents[2] / "Dados" / "data_for_ml.data"
df = pd.read_csv(data_path, sep=",", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
num_features = X.shape[1]
num_classes = len(set(y))

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
)


def objective(trial):
    # Definindo o espaço de busca de hiperparâmetros
    clauses = trial.suggest_int("num_clauses_per_class", 10, 500)
    T = trial.suggest_int("T", 5, 200)
    s = trial.suggest_float("s", 1.0, 20.0)

    epochs = 25

    model = TsetlinMachine(
        num_clauses_per_class=clauses,
        num_features=num_features,
        num_classes=num_classes,
        T=T,
        s=s,
    )

    model.fit(X_train, y_train, epochs=epochs)

    y_pred = model.predict(X_valid)
    accuracy = np.mean(y_pred == y_valid)

    # O optuna tenta porpadrão maximizar a saida dessa função
    return accuracy


print("Iniciando a busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("\n--- Resultados da Busca ---")
print(f"Melhor Acurácia na Validação: {study.best_value * 100:.2f}%")
print("Melhores Hiperparâmetros encontrados:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Retreinando com os melhores hiperparâmetros
print(
    "\nRetreinando o modelo final com os melhores hiperparâmetros (usando X_train_full)..."
)
best_model = TsetlinMachine(
    num_clauses_per_class=study.best_params["num_clauses_per_class"],
    num_features=num_features,
    num_classes=num_classes,
    T=study.best_params["T"],
    s=study.best_params["s"],
)

with open("parameters.txt", "w", encoding="utf-8") as f:
    f.write("=== Hiperparâmetros da Tsetlin Machine ===\n")
    f.write(
        f"Número de Cláusulas por Classe: {study.best_params['num_clauses_per_class']}\n"
    )
    f.write(f"Limiar de Votação (T): {study.best_params['T']}\n")
    f.write(f"Especificidade (s): {study.best_params['s']}\n")

# Treinando o modelo final sobre todo o dataset
best_model.fit(X_train_full, y_train_full, epochs=25)

best_model.save("model")

# Avaliação final no conjunto de Teste (dados nunca vistos pelo modelo ou pelo Optuna)
y_pred_test = best_model.predict(X_test)
test_accuracy = np.mean(y_pred_test == y_test) * 100
print(f"Acurácia Final no Conjunto de Teste: {test_accuracy:.2f}%")
