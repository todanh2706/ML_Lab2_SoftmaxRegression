# data.py
import numpy as np

def load_mnist_npz(path="mnist.npz"):
    """
    Expect file mnist.npz with keys: x_train, y_train, x_test, y_test.
    Shapes:
      x_*: (N, 28, 28)
      y_*: (N,)
    """
    data = np.load(path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return x_train, y_train, x_test, y_test

def train_val_split(x, y, val_ratio=0.1, shuffle=True, seed=42):
    N = x.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    x = x[idx]
    y = y[idx]

    n_val = int(N * val_ratio)
    x_val = x[:n_val]
    y_val = y[:n_val]
    x_train = x[n_val:]
    y_train = y[n_val:]

    return x_train, y_train, x_val, y_val
