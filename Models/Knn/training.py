import pandas as pd
from pathlib import Path
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier
import joblib  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_path = Path(__file__).resolve().parents[2] / "Dados" / "Divididos"
df_train = pd.read_csv(data_path / "train.csv", sep=",")
df_valid = pd.read_csv(data_path / "valid.csv", sep=",")
df_test = pd.read_csv(data_path / "test.csv", sep=",")

print("Shape dos dados:")
print("Treino:", df_train.shape)
print("Validação:", df_valid.shape)
print("Teste:", df_test.shape)

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_valid = df_valid.iloc[:, :-1].values
y_valid = df_valid.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values


num_features = X_train.shape[1]
num_classes = len(set(y_train))


def objective(trial):
    # Definindo o espaço de busca de hiperparâmetros para o KNN
    # n_neighbors: quantidade de vizinhos próximos a considerar
    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
    
    # weights: se todos os vizinhos têm peso igual ('uniform') 
    # ou se os mais próximos pesam mais ('distance')
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    
    # p: métrica de distância (1 = Manhattan, 2 = Euclidiana)
    p = trial.suggest_int("p", 1, 2)

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    accuracy = np.mean(y_pred == y_valid)

    # O optuna tenta por padrão maximizar a saida dessa função
    return accuracy


print("\nIniciando a busca de hiperparâmetros com Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\n--- Resultados da Busca ---")

print("\nRetreinando o modelo final com os melhores hiperparâmetros...")
best_model = KNeighborsClassifier(
    n_neighbors=study.best_params["n_neighbors"],
    weights=study.best_params["weights"],
    p=study.best_params["p"],
)

# Retreinando com os melhores parâmetros (novamente, sem epochs)
best_model.fit(X_train, y_train)

# Salvando o modelo usando joblib
joblib.dump(best_model, "best_knn_model.joblib")

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
