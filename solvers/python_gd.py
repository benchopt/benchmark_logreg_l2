from benchopt import BaseSolver

import numpy as np
from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = 1.9 / L

        X, y, lmbd = self.X, self.y, self.lmbd

        c = 0
        w = np.zeros(n_features)
        for i in range(n_iter):
            ywTx = y * (X @ w - c)
            temp = 1. / (1. + np.exp(ywTx))
            grad = -(X.T @ (y * temp)) + lmbd * w
            w -= step * grad
            if self.fit_intercept:
                c -= (y * temp).sum()

        self.w = w
        if self.fit_intercept:
            self.w = np.r_[self.w, c]

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2 / 4 + self.lmbd
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4 + self.lmbd
        return L
