"""Run binary BAOA to select features for a KNN classifier.

Objective: minimize 1-accuracy + lambda*(#selected/total).
"""

import os
import numpy as np
from binary_BAOA import binary_BAOA
from typing import Tuple, List
from collections import defaultdict
import matplotlib.pyplot as plt


def make_synthetic_dataset(n_train=60, n_test=30, dim=6, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    means = [np.zeros(dim), np.ones(dim) * 2.0]
    X_train = np.vstack([rng.randn(n_train // 2, dim) + means[0], rng.randn(n_train // 2, dim) + means[1]])
    y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))
    X_test = np.vstack([rng.randn(n_test // 2, dim) + means[0], rng.randn(n_test // 2, dim) + means[1]])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))
    return X_train, y_train, X_test, y_test


def load_breast_cancer_dataset(path_data: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Read CSV-like data, return X (n_samples, n_attributes), y (labels 0/1), and attribute names
    raw = []
    with open(path_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            raw.append(parts)

    # Attributes per names file: class + 9 attributes
    # We'll one-hot encode each attribute and keep attribute boundaries
    if len(raw) == 0:
        raise ValueError(f'No data found in {path_data}')

    data = np.array(raw, dtype=object)
    y_raw = data[:, 0].astype(str)
    y = (y_raw != 'no-recurrence-events').astype(int)

    attrs = data[:, 1:10]
    n_samples, n_attrs = attrs.shape

    # For each attribute collect unique categories in appearance order and map
    categories = []
    for j in range(n_attrs):
        seen = []
        for v in attrs[:, j]:
            if v == '?' or v in seen:
                continue
            seen.append(v)
        categories.append(seen)

    # Build one-hot columns and keep mapping from attribute -> column indices
    columns = []
    attr_to_cols = []
    col_counter = 0
    for j in range(n_attrs):
        cats = categories[j]
        cols = np.zeros((n_samples, len(cats)), dtype=float) if len(cats) > 0 else np.zeros((n_samples, 0), dtype=float)
        for i, val in enumerate(attrs[:, j]):
            if val == '?' or len(cats) == 0:
                # leave as zeros
                continue
            try:
                idx = cats.index(val)
            except ValueError:
                # unseen category (shouldn't happen) -> skip
                continue
            cols[i, idx] = 1.0

        if cols.shape[1] > 0:
            columns.append(cols)
            col_idxs = list(range(col_counter, col_counter + cols.shape[1]))
            attr_to_cols.append(col_idxs)
            col_counter += cols.shape[1]
        else:
            # attribute had no categories (unlikely) -> map to empty list
            attr_to_cols.append([])

    if len(columns) == 0:
        raise ValueError('No attribute columns were created from the dataset')

    X = np.hstack(columns)
    return X, y, attr_to_cols


def knn_predict(X_train, y_train, X_test, mask, k=3):
    if mask.sum() == 0:
        # avoid empty mask
        mask = np.ones_like(mask)
    Xtr = X_train[:, mask.astype(bool)]
    Xte = X_test[:, mask.astype(bool)]
    dists = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2))
    idx = np.argsort(dists, axis=1)[:, :k]
    preds = np.array([np.argmax(np.bincount(y_train[rows])) for rows in idx])
    return preds


def eval_mask(mask, X_train, y_train, X_test, y_test, lam=0.01):
    preds = knn_predict(X_train, y_train, X_test, mask)
    acc = float((preds == y_test).mean())
    penalty = lam * (mask.sum() / mask.size)
    return 1.0 - acc + penalty


def main():
    # If the breast-cancer.data file exists, load it and use attribute-level masks
    data_path = os.path.join(os.path.dirname(__file__), 'Breast Cancer Data', 'breast-cancer.data')
    if os.path.exists(data_path):
        X_all, y_all, attr_to_cols = load_breast_cancer_dataset(data_path)
        # simple stratified split 70/30
        n = X_all.shape[0]
        idx = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(idx)
        cut = int(0.7 * n)
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        # Number of attributes (not one-hot columns)
        n_bits = len(attr_to_cols)
        use_attr_level = True
    else:
        X_train, y_train, X_test, y_test = make_synthetic_dataset(n_train=80, n_test=40, dim=8, seed=2)
        n_bits = X_train.shape[1]
        attr_to_cols = [list(range(n_bits))]
        use_attr_level = False

    def objective(mask):
        # mask is at attribute level: expand to one-hot columns if needed
        if use_attr_level:
            # build column-level mask
            col_mask = np.zeros(X_train.shape[1], dtype=int)
            for ai, selected in enumerate(mask):
                if selected:
                    col_mask[attr_to_cols[ai]] = 1
            return eval_mask(col_mask, X_train, y_train, X_test, y_test, lam=0.01)
        else:
            return eval_mask(mask, X_train, y_train, X_test, y_test, lam=0.01)

    # Run binary BAOA
    history = {}
    agents = [20, 30, 50]
    iterations = [50, 100, 500]
    for n_agents in agents:
        for n_iter in iterations:
            best_mask, best_f, hist = binary_BAOA(objective, n_bits=n_bits, num_agents=n_agents, max_iter=n_iter, seed=3, transfer='sigmoid', return_history=True)
            history[(n_agents, n_iter)] = (best_mask, best_f, hist)
            print(f'Agents: {n_agents}, Iterations: {n_iter}, Best f: {best_f:.4f}, Selected: {int(best_mask.sum())}/{n_bits}')
            if use_attr_level:
                print('Best attribute mask:', best_mask)
                print('Selected attributes:', int(best_mask.sum()), '/', n_bits)
                # compute estimated accuracy by expanding mask
        col_mask = np.zeros(X_train.shape[1], dtype=int)
        for ai, selected in enumerate(best_mask):
            if selected:
                col_mask[attr_to_cols[ai]] = 1
        best_acc = 1.0 - eval_mask(col_mask, X_train, y_train, X_test, y_test, lam=0.0)

    print('Estimated accuracy (approx):', best_acc)
    num_plots = len(history)
    cols = 3                       # choose number of columns you want
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))

    for idx, ((n_agents, n_iter), hist) in enumerate(history.items(), 1):
        plt.subplot(rows, cols, idx)
        plt.plot(hist[2])
        plt.title(f'Agents: {n_agents}, Iterations: {n_iter}')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
