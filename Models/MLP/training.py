from MLP import MLP
import pandas as pd
from pathlib import Path
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_path = Path(__file__).resolve().parents[2] / "Dados" / "Divididos"
df_train = pd.read_csv(data_path / "train.csv", sep=",")
df_valid = pd.read_csv(data_path / "valid.csv", sep=",")
df_test = pd.read_csv(data_path / "test.csv", sep=",")

print(f"Shape Treino: {df_train.shape}")
print(f"Shape Validação: {df_valid.shape}")
print(f"Shape Teste: {df_test.shape}")


X_train = df_train.iloc[:, :-1].values.astype(np.float32)
y_train = df_train.iloc[:, -1].values.astype(np.int32)

X_valid = df_valid.iloc[:, :-1].values.astype(np.float32)
y_valid = df_valid.iloc[:, -1].values.astype(np.int32)

X_test = df_test.iloc[:, :-1].values.astype(np.float32)
y_test = df_test.iloc[:, -1].values.astype(np.int32)

num_features = X_train.shape[1]
num_classes = len(set(y_train))

def objective(trial):
    possible_topologies = [
        (8,),       
        (16,),     
        (8, 4),     
        (16, 8)     
    ]
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", possible_topologies)
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"])
    learning_rate_init = trial.suggest_categorical("learning_rate_init", [0.001, 0.01, 0.1, 0.5, 0.9])
    momentum = trial.suggest_categorical("momentum", [0.0, 0.1, 0.5, 0.9])
    max_iter = trial.suggest_categorical("max_iter", [200, 500, 1000])
    model = MLP(
        num_classes=num_classes,
        num_features=num_features,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        momentum=momentum,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    accuracy = np.mean(y_pred == y_valid)
    
    return accuracy


print("\noptuna hyperparameters discoverying")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\nresults")
print(f"best validation accuracy: {study.best_value * 100:.2f}%")


print("\ncheckpoint - retreining with bets hyperparameters")
best_model = MLP(
    num_classes=num_classes,
    num_features=num_features,
    hidden_layer_sizes=study.best_params["hidden_layer_sizes"],
    activation=study.best_params["activation"],
    solver=study.best_params["solver"],
    learning_rate_init=study.best_params["learning_rate_init"],
    max_iter=study.best_params["max_iter"],
    momentum=study.best_params["momentum"],
    random_state=42
)

best_model.fit(X_train, y_train)
best_model.save("model")

def calcular_metricas(modelo, X, y_true):
    y_pred = modelo.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

acc_train, prec_train, rec_train, f1_train = calcular_metricas(best_model, X_train, y_train)
acc_valid, prec_valid, rec_valid, f1_valid = calcular_metricas(best_model, X_valid, y_valid)
acc_test, prec_test, rec_test, f1_test = calcular_metricas(best_model, X_test, y_test)

out_file = "parameters.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("RESULTS\n")
    f.write("best hyperparameters (hyperparameters tunning)\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write("-" * 40 + "\n")
    
    f.write("TRAINING METRICS\n")
    f.write(f"Accuracy:  {acc_train * 100:.2f}%\n")
    f.write(f"Precision: {prec_train * 100:.2f}%\n")
    f.write(f"Recall:    {rec_train * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_train * 100:.2f}%\n")
    f.write("-" * 40 + "\n")
    
    f.write("VALIDATION METRICS \n")
    f.write(f"Accuracy:  {acc_valid * 100:.2f}%\n")
    f.write(f"Precision: {prec_valid * 100:.2f}%\n")
    f.write(f"Recall:    {rec_valid * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_valid * 100:.2f}%\n")
    f.write("-" * 40 + "\n")
    
    f.write("TEST METRICS \n")
    f.write(f"Accuracy:  {acc_test * 100:.2f}%\n")
    f.write(f"Precision: {prec_test * 100:.2f}%\n")
    f.write(f"Recall:    {rec_test * 100:.2f}%\n")
    f.write(f"F1-Score:  {f1_test * 100:.2f}%\n")
    f.write("-" * 40 + "\n")

print(f"saved '{out_file}'.")