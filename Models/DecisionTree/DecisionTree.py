import numpy as np
import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    

    def __init__(self, num_classes, num_features, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, criterion="gini"):
        """
        Inicializa a Árvore de Decisão.

        Parâmetros:
        -----------
        num_classes : int
            Número de classes
        num_features : int
            Número de features (não é usado diretamente, apenas para compatibilidade)
        max_depth : int, optional
            Profundidade máxima da árvore (default: 10)
        min_samples_split : int, optional
            Número mínimo de amostras para dividir um nó (default: 2)
        min_samples_leaf : int, optional
            Número mínimo de amostras em uma folha (default: 1)
        criterion : str, optional
            Função para medir a qualidade de uma divisão ('gini' ou 'entropy') (default: 'gini')
        """
        self.num_classes = num_classes
        self.num_features = num_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )

    def fit(self, X, y, epochs=1):
        """

        Parâmetros:
        -----------
        X : np.ndarray
            Features de treinamento de forma (n_samples, n_features)
        y : np.ndarray
            Labels de forma (n_samples,)
        epochs : int, optional
            Número de épocas (não é usado para Decision Tree, apenas para compatibilidade) (default: 1)
        """
        self.model.fit(X, y)
        
        #acurácia de treinamento
        y_pred = self.model.predict(X)
        train_accuracy = np.mean(y_pred == y) * 100
        print(f"Árvore de Decisão treinada - Acurácia de Treino: {train_accuracy:.2f}%")

    def predict(self, X):
        """

        Parâmetros:
        -----------
        X : np.ndarray
            Features de forma (n_samples, n_features)

        Retorna:
        --------
        np.ndarray
            Predições de forma (n_samples,)
        """
        return self.model.predict(X)

    def save(self, filepath):
       
        # Garante que a extensão seja .pkl
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"

        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "num_classes": self.num_classes,
                "num_features": self.num_features,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
            }, f)
        print(f"Modelo salvo com sucesso em: {filepath}")

    @classmethod
    def load(cls, filepath):
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model_instance = cls(
            num_classes=data["num_classes"],
            num_features=data["num_features"],
            max_depth=data["max_depth"],
            min_samples_split=data["min_samples_split"],
            min_samples_leaf=data["min_samples_leaf"],
            criterion=data["criterion"],
        )

        model_instance.model = data["model"]
        print(f"Modelo carregado com sucesso de: {filepath}")
        return model_instance
