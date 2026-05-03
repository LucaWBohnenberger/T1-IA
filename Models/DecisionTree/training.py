from DecisionTree import DecisionTree
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


def objective(trial):
   
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    model = DecisionTree(
        num_classes=num_classes,
        num_features=num_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    accuracy = np.mean(y_pred == y_valid)

    
    return accuracy


print("Iniciando a busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n--- Resultados da Busca ---")

print("\nRetreinando o modelo final com os melhores hiperparâmetros...")
best_model = DecisionTree(
    num_classes=num_classes,
    num_features=num_features,
    max_depth=study.best_params["max_depth"],
    min_samples_split=study.best_params["min_samples_split"],
    min_samples_leaf=study.best_params["min_samples_leaf"],
    criterion=study.best_params["criterion"],
)

best_model.fit(X_train, y_train)
best_model.save("model")
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

