import pandas as pd
from pathlib import Path
import numpy as np
import optuna
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Carregamento e Preparação dos Dados
data_path = Path(__file__).resolve().parents[2] / "Dados" / "Divididos"
df_train = pd.read_csv(data_path / "train.csv", sep=",")
df_valid = pd.read_csv(data_path / "valid.csv", sep=",")
df_test = pd.read_csv(data_path / "test.csv", sep=",")

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_valid = df_valid.iloc[:, :-1].values
y_valid = df_valid.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values


def objective(trial):
    # C: Parâmetro de penalidade. 
    # Como nosso problema é deterministico, um C alto pode funcionar melhor, mas vou deixar o optuna decidir.
    C = trial.suggest_float("C", 0.1, 1000.0, log=True)
    
    # kernel: A forma matemática da fronteira de decisão
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    
    # gamma: Relevante apenas para kernels não-lineares (rbf e poly). 
    # Define o alcance da influência de um único exemplo de treinamento.
    if kernel in ["rbf", "poly"]:
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    else:
        gamma = "scale" # Ignorado no kernel linear
        
    # degree: Grau do polinômio (relevante apenas se o kernel for 'poly')
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3 # Padrão ignorado nos outros kernels

    model = SVC(
        C=C, 
        kernel=kernel, 
        gamma=gamma, 
        degree=degree,
        random_state=42 
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    accuracy = np.mean(y_pred == y_valid)

    return accuracy

print("\nIniciando a busca de hiperparâmetros com Optuna para SVM...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\nRetreinando o modelo final com os melhores hiperparâmetros...")

best_kernel = study.best_params["kernel"]
best_model = SVC(
    C=study.best_params["C"],
    kernel=best_kernel,
    gamma=study.best_params.get("gamma", "scale"),
    degree=study.best_params.get("degree", 3),
    random_state=42
)

best_model.fit(X_train, y_train)
joblib.dump(best_model, "best_svm_model.joblib")



# Calculando previsões e métricas para TREINO
y_pred_train = best_model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
prec_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

# Calculando previsões e métricas para VALIDAÇÃO
y_pred_valid = best_model.predict(X_valid)
acc_valid = accuracy_score(y_valid, y_pred_valid)
prec_valid = precision_score(y_valid, y_pred_valid, average='weighted', zero_division=0)
rec_valid = recall_score(y_valid, y_pred_valid, average='weighted', zero_division=0)
f1_valid = f1_score(y_valid, y_pred_valid, average='weighted', zero_division=0)

# Calculando previsões e métricas para TESTE
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

print("\nBusca e avaliação finalizadas!")
print("As 4 métricas para Treino, Validação e Teste foram salvas no arquivo 'parameters.txt'.")
