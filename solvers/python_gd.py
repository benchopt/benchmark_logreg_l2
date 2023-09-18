from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = 1. / L

        X, y, lmbd = self.X, self.y, self.lmbd

        w = np.zeros(n_features)
        for i in range(n_iter):
            ywTx = y * (X @ w)
            temp = 1. / (1. + np.exp(ywTx))
            grad = -(X.T @ (y * temp)) + lmbd * w
            w -= step * grad

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2 / 4 + self.lmbd
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4 + self.lmbd
        return L
