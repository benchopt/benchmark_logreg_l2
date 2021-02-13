import numpy as np


from benchopt.base import BaseSolver


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        n_samples, n_features = self.X.shape

        L = (np.linalg.norm(self.X) ** 2 / 4) + self.lmbd
        step = 1. / L

        X, y, lmbd = self.X, self.y, self.lmbd

        w = np.zeros(n_features)
        for i in range(n_iter):
            ywTx = y * np.dot(X, w)
            temp = 1. / (1. + np.exp(ywTx))
            grad = -np.dot(X.T, (y * temp)) + lmbd * w
            w -= step * grad

        self.w = w

    def get_result(self):
        return self.w
