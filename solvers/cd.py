from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numba import njit


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


class Solver(BaseSolver):
    name = "cd"

    install_cmd = 'conda'
    requirements = ['numba']

    def set_objective(self, X, y, lmbd):
        self.y, self.lmbd = y, lmbd

        if sparse.issparse(X):
            self.X = X
        else:
            # use Fortran order to compute gradient on contiguous columns
            self.X = np.asfortranarray(X)

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        if sparse.issparse(self.X):
            L = np.array((self.X.multiply(self.X)).sum(axis=0)).squeeze() / 4
            L += self.lmbd
            self.w = self.sparse_cd(
                self.X.data, self.X.indices, self.X.indptr, self.y, self.lmbd,
                L, n_iter
            )
        else:
            L = (self.X ** 2).sum(axis=0) / 4
            L += self.lmbd
            self.w = self.cd(self.X, self.y, self.lmbd, L, n_iter)

    @staticmethod
    @njit
    def cd(X, y, lmbd, L, n_iter):
        n_features = X.shape[1]
        Xw = np.zeros_like(y)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                grad_j = np.sum(-y * X[:, j] / (1 + np.exp(y * Xw)))
                grad_j += lmbd * w[j]
                w[j] -= grad_j / L[j]
                diff = w[j] - old
                if diff != 0:
                    Xw += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def sparse_cd(X_data, X_indices, X_indptr, y, lmbd, L, n_iter):
        n_features = len(X_indptr) - 1
        Xw = np.zeros_like(y)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                start, end = X_indptr[j:j+2]
                scal = lmbd * w[j]
                for ind in range(start, end):
                    scal += (
                        -y[X_indices[ind]] * X_data[ind] /
                        (1 + np.exp(y[X_indices[ind]] * Xw[X_indices[ind]]))
                    )
                w[j] -= scal / L[j]
                diff = w[j] - old
                if diff != 0:
                    for ind in range(start, end):
                        Xw[X_indices[ind]] += diff * X_data[ind]
        return w

    def get_result(self):
        return self.w
