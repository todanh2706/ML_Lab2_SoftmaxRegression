# softmax_model.py
import numpy as np

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, lr=0.1, reg=0.0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.reg = reg

        # Khởi tạo tham số
        self.W = 0.01 * np.random.randn(n_classes, n_features).astype(np.float32)
        self.b = np.zeros((n_classes,), dtype=np.float32)

    # --------- softmax + utils ---------
    @staticmethod
    def softmax(Z):
        """
        Z: (N, K)
        return: (N, K) probabilities
        """
        # Numerical Stability
        Z_shift = Z - Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def one_hot(y, num_classes):
        N = y.shape[0]
        Y = np.zeros((N, num_classes), dtype=np.float32)
        Y[np.arange(N), y] = 1.0
        return Y

    # --------- forward + loss + gradient ---------
    def compute_loss_and_grads(self, X, y):
        """
        X: (N, d), y: (N,)
        """
        N = X.shape[0]
        # logits
        scores = X @ self.W.T + self.b  # (N, K)
        probs = self.softmax(scores)    # (N, K)

        # cross-entropy loss
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-15)
        data_loss = correct_logprobs.mean()
        reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
        loss = data_loss + reg_loss

        # gradient
        dscores = probs
        dscores[np.arange(N), y] -= 1  # (N, K)
        dscores /= N

        dW = dscores.T @ X  # (K, d)
        if self.reg > 0:
            dW += self.reg * self.W

        db = dscores.sum(axis=0)  # (K,)

        return loss, dW, db

    # --------- train (gradient descent) ---------
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=20, batch_size=128, verbose=True):

        N = X_train.shape[0]
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # shuffle
            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                loss, dW, db = self.compute_loss_and_grads(X_batch, y_batch)
                epoch_loss += loss
                n_batches += 1

                # gradient descent step
                self.W -= self.lr * dW
                self.b -= self.lr * db

            avg_loss = epoch_loss / max(n_batches, 1)

            val_acc = None
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                val_acc = (y_pred == y_val).mean()
                history["val_acc"].append(val_acc)

            history["train_loss"].append(avg_loss)

            if verbose:
                if val_acc is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}, val_acc={val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")

        return history

    # --------- predict ---------
    def predict_proba(self, X):
        # Model hypothesis
        scores = X @ self.W.T + self.b
        probs = self.softmax(scores)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # --------- save / load ---------
    def save(self, path):
        np.savez(path, W=self.W, b=self.b,
                 n_features=self.n_features, n_classes=self.n_classes,
                 lr=self.lr, reg=self.reg)

    @staticmethod
    def load(path):
        data = np.load(path)
        n_features = int(data["n_features"])
        n_classes = int(data["n_classes"])
        lr = float(data["lr"])
        reg = float(data["reg"])
        model = SoftmaxRegression(n_features, n_classes, lr=lr, reg=reg)
        model.W = data["W"]
        model.b = data["b"]
        return model
