import numpy as np


class TsetlinMachine:
    def __init__(
        self, num_classes, num_features, num_clauses_per_class, T, s, num_states=256
    ):
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_clauses = num_clauses_per_class
        self.T = T
        self.s = s
        self.num_states = num_states
        self.threshold = self.num_states // 2

        self.ta_states = np.random.choice(
            [self.threshold, self.threshold + 1],
            size=(self.num_classes, self.num_clauses, 2 * self.num_features),
        ).astype(np.int16)

    def _get_literals(self, X):
        return np.concatenate((X, 1 - X), axis=1).astype(bool)

    def _get_clause_outputs(self, X_literals):
        ta_action_include = self.ta_states > self.threshold
        X_expanded = np.expand_dims(np.expand_dims(X_literals, 1), 1)
        return np.all(X_expanded | (~ta_action_include), axis=3)

    def predict(self, X):
        X_literals = self._get_literals(X)
        clause_outputs = self._get_clause_outputs(X_literals)
        pos_clauses = clause_outputs[:, :, 0::2]
        neg_clauses = clause_outputs[:, :, 1::2]
        class_scores = np.sum(pos_clauses, axis=2) - np.sum(neg_clauses, axis=2)
        return np.argmax(class_scores, axis=1)
    
    def fit(self, X, y, epochs=10):
        """Treina o modelo usando Stochastic Gradient Descent (SGD) style"""
        num_samples = len(X)

        for epoch in range(epochs):
            # Embaralha os dados a cada época
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            erros = 0

            for i in indices:
                x_sample = X[i : i + 1]
                y_true = y[i]
                
                x_lit = self._get_literals(x_sample)[0]  # (2 * features,)
                c_out = self._get_clause_outputs(self._get_literals(x_sample))[
                    0
                ]  # (classes, clauses)


                pos_c = c_out[:, 0::2]
                neg_c = c_out[:, 1::2]
                scores = np.clip(
                    np.sum(pos_c, axis=1) - np.sum(neg_c, axis=1), -self.T, self.T
                )

                if np.argmax(scores) != y_true:
                    erros += 1

       
                neg_classes = [c for c in range(self.num_classes) if c != y_true]
                y_neg = np.random.choice(neg_classes)

      
                prob_true = (self.T - scores[y_true]) / (2 * self.T)
                prob_neg = (self.T + scores[y_neg]) / (2 * self.T)


                self._apply_feedback(
                    y_true, x_lit, c_out[y_true], prob_true, is_target_class=True
                )

                self._apply_feedback(
                    y_neg, x_lit, c_out[y_neg], prob_neg, is_target_class=False
                )

            print(
                f"Época {epoch + 1}/{epochs} - Acurácia de Treino aprox: {100 - (erros / num_samples) * 100:.2f}%"
            )

    def _apply_feedback(self, class_idx, x_lit, c_out, prob, is_target_class):
        """Direciona o Tipo I e Tipo II dependendo da polaridade da cláusula"""
        states = self.ta_states[class_idx]


        update_mask = np.random.rand(self.num_clauses) < prob

        for c in range(self.num_clauses):
            if not update_mask[c]:
                continue

            is_pos_clause = c % 2 == 0

            # Lógica de Roteamento de Feedback
            if is_target_class:
                if is_pos_clause:
                    self._type_1_feedback(states[c], x_lit, c_out[c])
                else:
                    self._type_2_feedback(states[c], x_lit, c_out[c])
            else:
                if is_pos_clause:
                    self._type_2_feedback(states[c], x_lit, c_out[c])
                else:
                    self._type_1_feedback(states[c], x_lit, c_out[c])

        # Garante que os estados não ultrapassem os limites da memória do autômato
        self.ta_states[class_idx] = np.clip(states, 1, self.num_states)

    def _type_1_feedback(self, state, x_lit, c_out):
        """Reforça padrões verdadeiros, penaliza ruído. Controlado pela especificidade 's'."""
        if c_out == 1:
            inc_mask = (np.random.rand(len(x_lit)) < (self.s - 1) / self.s) & x_lit
            dec_mask = (np.random.rand(len(x_lit)) < 1 / self.s) & (~x_lit)
            state += inc_mask
            state -= dec_mask
        else:
            dec_mask = np.random.rand(len(x_lit)) < 1 / self.s
            state -= dec_mask

    def _type_2_feedback(self, state, x_lit, c_out):
        """Força a cláusula a ser Falsa se ela avaliou como Verdadeira erroneamente."""
        if c_out == 1:
            inc_mask = ~x_lit  # Se a feature é 0, inclui ela para quebrar o AND lógico
            state += inc_mask

    # ==========================================
    # SALVAR E CARREGAR MODELO
    # ==========================================

    def save(self, filepath):
        """
        Salva os hiperparâmetros e a matriz de estados do modelo em um arquivo .npz.

        Parâmetros:
        filepath (str): O caminho ou nome do arquivo onde o modelo será salvo (ex: 'meu_modelo.npz').
        """
        # Garante que a extensão seja .npz
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        np.savez_compressed(
            filepath,
            ta_states=self.ta_states,
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_clauses=self.num_clauses,
            T=self.T,
            s=self.s,
            num_states=self.num_states,
        )
        print(f"Modelo salvo com sucesso em: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Lê um arquivo .npz e reconstrói a TsetlinMachine com os estados salvos.

        Parâmetros:
        filepath (str): O caminho do arquivo salvo.

        Retorna:
        TsetlinMachine: Uma nova instância do modelo configurada e com os estados carregados.
        """

        data = np.load(filepath)

        model = cls(
            num_classes=int(data["num_classes"]),
            num_features=int(data["num_features"]),
            num_clauses_per_class=int(data["num_clauses"]),
            T=int(data["T"]),
            s=float(data["s"]),
            num_states=int(data["num_states"]),
        )

        model.ta_states = data["ta_states"]

        print(f"Modelo carregado com sucesso de: {filepath}")
        return model
