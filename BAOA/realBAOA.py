# knn_baoa.py
# Requires: numpy, scikit-learn
# pip install numpy scikit-learn

import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import random
import time

def load_breast_cancer_dataset(path_data: str):
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


# -----------------------------
# Helper: evaluate a binary mask with KNN (cross-validated accuracy)
# -----------------------------
def evaluate_mask(X, y, mask, knn=None, cv=5, scoring='accuracy', random_state=None):
    if knn is None:
        knn = KNeighborsClassifier(n_neighbors=5)
    # if no features selected -> return poor score
    if np.sum(mask) == 0:
        return 0.0
    Xs = X[:, mask == 1]
    # use stratified kfold
    cvsplit = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(knn, Xs, y, cv=cvsplit, scoring=scoring)
    return np.mean(scores)

# -----------------------------
# BAOA: Binary Arithmetic Optimization Algorithm for Feature Selection
# (a practical implementation inspired by BAOA literature)
# -----------------------------
class BAOAFeatureSelector:
    def __init__(self,
                 n_agents=30,
                 max_iters=100,
                 alpha=0.99,          # controls math operator probability schedule
                 w_acc=0.9,           # weight for accuracy in objective (higher -> prefer accuracy)
                 w_feat=0.1,          # weight for feature subset size (smaller -> better)
                 knn=None,
                 cv=5,
                 random_state=None):
        self.n_agents = n_agents
        self.max_iters = max_iters
        self.alpha = alpha
        self.w_acc = w_acc
        self.w_feat = w_feat
        self.knn = knn if knn is not None else KNeighborsClassifier(n_neighbors=5)
        self.cv = cv
        self.rng = np.random.RandomState(random_state)
        self.best_mask = None
        self.best_score = -np.inf
        self.history = []

    def _init_population(self, dim):
        # continuous position in [-6,6] for each feature to allow varied transfer to binary
        pop = self.rng.uniform(-6, 6, (self.n_agents, dim))
        return pop

    @staticmethod
    def _sigmoid(x):
        # transfer function: map real value to probability of selecting 1
        return 1.0 / (1.0 + np.exp(-x))

    def _binary_from_continuous(self, cont_pos):
        # cont_pos: (n_agents, dim) -> binary masks (0/1)
        probs = self._sigmoid(cont_pos)
        rand = self.rng.rand(*probs.shape)
        return (probs > rand).astype(int)

    def _fitness(self, X, y, mask):
        # objective: combine accuracy and compactness
        acc = evaluate_mask(X, y, mask, knn=self.knn, cv=self.cv, random_state=None)
        feat_ratio = (np.sum(mask) / mask.size)
        # we want to maximize accuracy and minimize features -> combine linearly
        score = self.w_acc * acc - self.w_feat * feat_ratio
        return score, acc

    def fit(self, X, y, verbose=True):
        n_features = X.shape[1]
        pop = self._init_population(n_features)          # continuous positions
        # leader = best agent (continuous)
        leader_pos = pop[0].copy()
        leader_score = -np.inf

        # evaluate initial population
        masks = self._binary_from_continuous(pop)
        scores = np.zeros(self.n_agents)
        accuracies = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            scores[i], accuracies[i] = self._fitness(X, y, masks[i])
            if scores[i] > leader_score:
                leader_score = scores[i]
                leader_pos = pop[i].copy()

        # main loop
        start_time = time.time()
        for t in range(self.max_iters):
            T = float(t) / self.max_iters  # normalized iteration
            # adaptive parameters borrowed from AOA idea:
            r1 = self.rng.rand(self.n_agents, n_features)
            # Math operator probability (to switch exploration/exploitation)
            MOA = self.alpha * (1 - (t / self.max_iters))  # decreases over time
            # Math optimizer probability (map to using multiplication/division vs add/sub)
            MOP = 0.5 * (1 + np.tanh((t - self.max_iters/2) / (self.max_iters/10)))

            for i in range(self.n_agents):
                for d in range(n_features):
                    if self.rng.rand() < MOA:
                        # exploration (use multiplication/division style update)
                        if self.rng.rand() < 0.5:
                            pop[i, d] = pop[i, d] * (1 + self.rng.randn() * 0.1)  # mutate
                        else:
                            pop[i, d] = pop[i, d] / (1 + self.rng.randn() * 0.1 + 1e-9)
                    else:
                        # exploitation: move towards leader using add/sub patterns
                        epsilon = self.rng.randn() * 0.1
                        pop[i, d] = pop[i, d] + epsilon * (leader_pos[d] - pop[i, d])

                # small random local search occasionally
                if self.rng.rand() < 0.1:
                    pop[i] += self.rng.normal(0, 0.05, size=n_features)

            # Clip continuous positions to reasonable range to stabilize sigmoid
            pop = np.clip(pop, -10, 10)

            # Evaluate
            masks = self._binary_from_continuous(pop)
            for i in range(self.n_agents):
                scores[i], accuracies[i] = self._fitness(X, y, masks[i])
                if scores[i] > leader_score:
                    leader_score = scores[i]
                    leader_pos = pop[i].copy()

            # record best
            best_idx = np.argmax(scores)
            current_best_score = scores[best_idx]
            current_best_acc = accuracies[best_idx]
            current_best_mask = masks[best_idx]
            self.history.append((current_best_score, current_best_acc, int(np.sum(current_best_mask))))
            if verbose and (t % max(1, self.max_iters // 10) == 0 or t == self.max_iters-1):
                print(f"[{t+1}/{self.max_iters}] best_score={current_best_score:.4f} acc={current_best_acc:.4f} selected={np.sum(current_best_mask)}")

        # save final best
        final_mask = (self._sigmoid(leader_pos) >  self.rng.rand(n_features)).astype(int)
        final_score, final_acc = self._fitness(X, y, final_mask)
        self.best_mask = final_mask
        self.best_score = final_score
        end_time = time.time()
        if verbose:
            print(f"Finished. Best score={final_score:.4f}, acc={final_acc:.4f}, selected_features={np.sum(final_mask)}, time={end_time-start_time:.2f}s")
        return self

    def transform(self, X):
        if self.best_mask is None:
            raise ValueError("Model not fitted yet.")
        return X[:, self.best_mask == 1]

    def fit_transform(self, X, y, verbose=True):
        self.fit(X, y, verbose=verbose)
        return self.transform(X)

# -----------------------------
# Example usage on breast cancer dataset
# -----------------------------
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'Breast Cancer Data', 'breast-cancer.data')
    X, y, attr_to_cols = load_breast_cancer_dataset(data_path)
    # standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    selector = BAOAFeatureSelector(n_agents=30, max_iters=80, w_acc=0.95, w_feat=0.05, cv=5, random_state=42)
    selector.fit(Xs, y, verbose=True)

    mask = selector.best_mask
    chosen_indices = np.where(mask == 1)[0]
    print("\nSelected feature indices (0-based):", chosen_indices)
    print("Number selected:", len(chosen_indices))

    # final evaluation of KNN on the selected features
    knn = KNeighborsClassifier(n_neighbors=5)
    final_scores = cross_val_score(knn, Xs[:, mask==1], y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print("Final CV accuracy on selected features: %.4f +/- %.4f" % (final_scores.mean(), final_scores.std()))
