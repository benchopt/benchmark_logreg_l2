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

        L = self.compute_lipschitz_cste()
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
        return self.w

    def compute_lipschitz_cste(self, max_iter=100):
        if not sparse.issparse(self.X):
            return np.linalg.norm(self.X, ord=2) ** 2 / 4 + self.lmbd

        n, m = self.X.shape
        if n < m:
            A = self.X.T
        else:
            A = self.X

        b_k = np.random.rand(A.shape[1])
        b_k /= np.linalg.norm(b_k)
        rk = np.inf

        for _ in range(max_iter):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T @ (A @ b_k)

            # compute the eigenvalue and stop if it does not move anymore
            rk1 = rk
            rk = b_k1 @ b_k
            if abs(rk - rk1) < 1e-10:
                break

            # re normalize the vector
            b_k = b_k1 / np.linalg.norm(b_k1)

        return rk / 4 + self.lmbd
