from TsetlinMachine import TsetlinMachine
import pandas as pd
from pathlib import Path
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


# Optuna objective function para otimização dos hiperparâmetros do Tsetlin Machine, optuna funciona tentando maximizar a acurácia de validação, então a função retorna a acurácia calculada no conjunto de validação. Ele usa uma estratégia de busca inteligente baseada em logica de máxima posteriori.
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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Calculando previsões e métricas para TREINO
y_pred_train = best_model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
prec_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

# 2. Calculando previsões e métricas para VALIDAÇÃO
y_pred_valid = best_model.predict(X_valid)
acc_valid = accuracy_score(y_valid, y_pred_valid)
prec_valid = precision_score(y_valid, y_pred_valid, average='weighted', zero_division=0)
rec_valid = recall_score(y_valid, y_pred_valid, average='weighted', zero_division=0)
f1_valid = f1_score(y_valid, y_pred_valid, average='weighted', zero_division=0)

# 3. Calculando previsões e métricas para TESTE
y_pred_test = best_model.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

# 4. Salvando tudo no arquivo texto
with open("parameters.txt", "w", encoding="utf-8") as f:
    f.write("=== RESULTADOS DO MODELO ===\n")
    f.write("Melhores Hiperparâmetros:\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write("-" * 40 + "\n")
    
    f.write("--- MÉTRICAS DE TREINAMENTO ---\n")
    f.write(f"Acurácia:  {acc_train * 100:.2f}%\n")
    f.write(f"Precision: {prec_train * 100:.2f}%\n")
    f.write(f"Recall:    {rec_train * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_train * 100:.2f}%\n")
    f.write("-" * 40 + "\n")
    
    f.write("--- MÉTRICAS DE VALIDAÇÃO ---\n")
    f.write(f"Acurácia:  {acc_valid * 100:.2f}%\n")
    f.write(f"Precision: {prec_valid * 100:.2f}%\n")
    f.write(f"Recall:    {rec_valid * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_valid * 100:.2f}%\n")
    f.write("-" * 40 + "\n")
    
    f.write("--- MÉTRICAS DE TESTE (DADOS INÉDITOS) ---\n")
    f.write(f"Acurácia:  {acc_test * 100:.2f}%\n")
    f.write(f"Precision: {prec_test * 100:.2f}%\n")
    f.write(f"Recall:    {rec_test * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_test * 100:.2f}%\n")
    f.write("-" * 40 + "\n")

print("Os resultados foram salvos em 'parameters.txt'.")
