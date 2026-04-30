from TsetlinMachine import TsetlinMachine
import pandas as pd
from pathlib import Path
import numpy as np
import optuna

# 1. Carregamento e Preparação dos Dados
data_path = Path(__file__).resolve().parents[2] / "Dados" / "Divididos"
df_train = pd.read_csv(data_path / "train.csv", sep=",")
df_valid = pd.read_csv(data_path / "valid.csv", sep=",")
df_test = pd.read_csv(data_path / "test.csv", sep=",")

print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_valid = df_valid.iloc[:, :-1].values
y_valid = df_valid.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

X_train = X_train.astype(np.int32)
X_valid = X_valid.astype(np.int32)
X_test = X_test.astype(np.int32)

num_features = X_train.shape[1]
num_classes = len(set(y_train))


def objective(trial):
    # Definindo o espaço de busca de hiperparâmetros
    clauses = trial.suggest_int("num_clauses_per_class", 10, 500)
    T = trial.suggest_int("T", 5, 200)
    s = trial.suggest_float("s", 1.0, 20.0)

    epochs = 30

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

    # O optuna tenta por padrão maximizar a saida dessa função
    return accuracy


print("Iniciando a busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n--- Resultados da Busca ---")

print("\nRetreinando o modelo final com os melhores hiperparâmetros...")
best_model = TsetlinMachine(
    num_clauses_per_class=study.best_params["num_clauses_per_class"],
    num_features=num_features,
    num_classes=num_classes,
    T=study.best_params["T"],
    s=study.best_params["s"],
)

best_model.fit(X_train, y_train, epochs=30)
best_model.save("model")

y_pred_test = best_model.predict(X_test)
test_accuracy = np.mean(y_pred_test == y_test) * 100

with open("parameters.txt", "w", encoding="utf-8") as f:
    f.write("=== RESULTADOS DO MODELO TSETLIN MACHINE ===\n")
    f.write(
        f"Número de Cláusulas por Classe: {study.best_params['num_clauses_per_class']}\n"
    )
    f.write(f"Limiar de Votação (T): {study.best_params['T']}\n")
    f.write(f"Especificidade (s): {study.best_params['s']}\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Acurácia de Validação (Melhor Trial Optuna): {study.best_value * 100:.2f}%\n"
    )
    f.write(f"Acurácia de Teste (Dados Inéditos): {test_accuracy:.2f}%\n")
    f.write("-" * 40 + "\n")

print(f"\nBusca finalizada!")
print("Melhores Hiperparâmetros encontrados:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Acurácia de Validação: {study.best_value * 100:.2f}%")
print(f"Acurácia de Teste: {test_accuracy:.2f}%")
print("Os resultados foram salvos em 'parameters.txt'.")
