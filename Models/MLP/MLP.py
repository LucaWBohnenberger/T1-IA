import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

class MLP:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 solver='adam', learning_rate_init=0.001, max_iter=200, 
                 momentum=0.9, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.momentum = momentum

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,      
            solver=solver,              
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            momentum=momentum,         
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        
        y_pred = self.model.predict(X)
        train_accuracy = np.mean(y_pred == y) * 100
        print(f"training accuracy= {train_accuracy:.2f}%")

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"

        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": self.activation,
                "solver": self.solver,
                "learning_rate_init": self.learning_rate_init,
                "max_iter": self.max_iter,
                "momentum": self.momentum,
            }, f)
        print(f"model saved {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        model_instance = cls(
            hidden_layer_sizes=data["hidden_layer_sizes"],
            activation=data["activation"],
            solver=data["solver"],
            learning_rate_init=data["learning_rate_init"],
            max_iter=data["max_iter"],
            momentum=data["momentum"]
        )

        model_instance.model = data["model"]
        print(f"model loaded {filepath}")
        return model_instance