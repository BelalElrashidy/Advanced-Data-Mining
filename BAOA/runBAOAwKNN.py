"""Run BAOA to optimize feature weights for a simple KNN classifier.

This script creates a tiny synthetic classification dataset, defines a KNN
accuracy evaluator that applies a non-negative weight vector to features, and
uses `BAOA` to search for weights that maximize accuracy (we minimize 1-accuracy).
"""

import numpy as np
from AOA import AOA
from typing import Tuple
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 


def read_uci_winequality_red(n_train=0.8, n_test=0.2, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    breast_cancer = fetch_ucirepo(id=14) 
    X = breast_cancer.data.features 
    y = breast_cancer.data.targets 
    # convert to binary classification: quality >= 6 -> 1, else 0
    y_binary = (y >= 6).astype(int)
    # split into train/test
    n_samples = X.shape[0]
    n_train = int(n_train * n_samples)
    n_test = int(n_test * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    X_train, y_train = X[train_idx], y_binary[train_idx]
    X_test, y_test = X[test_idx], y_binary[test_idx]
    return X_train, y_train, X_test, y_test



def make_dataset(n_train=60, n_test=30, dim=4, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    # two classes, gaussian clouds
    means = [np.zeros(dim), np.ones(dim) * 2.0]
    X_train = np.vstack([rng.randn(n_train // 2, dim) + means[0], rng.randn(n_train // 2, dim) + means[1]])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))
    X_test = np.vstack([rng.randn(n_test // 2, dim) + means[0], rng.randn(n_test // 2, dim) + means[1]])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))
    return X_train, y_train, X_test, y_test


def knn_predict(X_train, y_train, X_test, weights, k=3):
    # weights should be non-negative; normalize
    w = np.maximum(weights, 0.0)
    if np.sum(w) == 0:
        w = np.ones_like(w)
    w = w / np.sum(w)
    # weighted euclidean distances
    dists = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2) * w[None, None, :]).sum(axis=2)
    idx = np.argsort(dists, axis=1)[:, :k]
    preds = np.array([np.argmax(np.bincount(y_train[rows])) for rows in idx])
    return preds


def eval_weights(weights, X_train, y_train, X_test, y_test):
    preds = knn_predict(X_train, y_train, X_test, weights)
    acc = float((preds == y_test).mean())
    # BAOA minimizes, so return 1-accuracy
    return 1.0 - acc


def main():
    X_train, y_train, X_test, y_test = read_uci_winequality_red()
    dim = X_train.shape[1]
    bounds = [(0.0, 5.0)] * dim  # feature weights between 0 and 5
    agents = [20, 30, 50]  # different agent counts to try
    iterations = [50, 100, 500]  # different iteration counts to try
    history = []
    h = {}
    def objective(w):
        return eval_weights(w, X_train, y_train, X_test, y_test)
    for n_agents in agents:
        for n_iter in iterations:
            best_w, best_f, hist = AOA(objective, bounds, num_agents=n_agents, max_iter=n_iter, seed=2, return_history=True)
            best_acc = 1.0 - best_f
            h[(n_agents, n_iter)] = (hist)
            history.append((n_agents, n_iter, best_w, best_acc, hist))

    # Find the best configuration
    
    best_config = max(history, key=lambda x: x[3])  # maximize accuracy
    print("Best configuration:")
    print("  Agents:", best_config[0])
    print("  Iterations:", best_config[1])
    print("  Weights:", best_config[2])
    print("  Accuracy:", best_config[3])
    # Plot the optimization history
    plt.figure(figsize=(10, 5))
    for (n_agents, n_iter), tup in h.items():
        # tup == (best_mask, best_f, hist)
        _, _, hist = tup
        plt.plot(hist, label=f'Agents: {n_agents}, Iterations: {n_iter}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('BAOA Optimization History')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
