from benchopt import BaseSolver

import math
import numpy as np
from scipy import sparse
from numba import njit


@njit
def _newton_step_size(X, exp_yXw, j, lmbd):
    hess_jj = 0.
    for i in range(len(X)):
        hess_jj += X[i, j]**2 * exp_yXw[i] / (1 + exp_yXw[i])**2
    return 1 / (hess_jj + lmbd)


@njit
def _newton_step_size_sparse(X_data, X_indices, X_indptr, exp_yXw, j, lmbd):
    start, end = X_indptr[j:j+2]
    hess_jj = 0.
    for ind in range(start, end):
        i = X_indices[ind]
        hess_jj += X_data[ind]**2 * exp_yXw[i] / (1 + exp_yXw[i])**2
    return 1 / (hess_jj + lmbd)


class Solver(BaseSolver):
    name = "cd"

    install_cmd = 'conda'
    requirements = ['numba']

    parameters = {
        'newton_step': [False, True],
    }

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, "no implemented with fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.y, self.lmbd = y, lmbd

        if sparse.issparse(X):
            self.X = X
        else:
            # use Fortran order to compute gradient on contiguous columns
            self.X = np.asfortranarray(X)

        # Make sure we cache the numba compilation.
        self.run(1)

    def _get_lipschitz_csts(self):
        if sparse.issparse(self.X):
            L = sparse.linalg.norm(self.X, axis=0)**2 / 4
        else:
            L = (self.X ** 2).sum(axis=0) / 4
        L += self.lmbd
        return L

    def run(self, n_iter):
        L = self._get_lipschitz_csts()
        if sparse.issparse(self.X):
            self.w = self.sparse_cd(
                self.X.data, self.X.indices, self.X.indptr, self.y, self.lmbd,
                L, n_iter, self.newton_step
            )
        else:
            self.w = self.cd(
                self.X, self.y, self.lmbd, L, n_iter, self.newton_step
            )

    @staticmethod
    @njit
    def cd(X, y, lmbd, L, n_iter, newton_step):
        n_samples, n_features = X.shape
        Xw = np.zeros(y.shape, dtype=X.dtype)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                exp_yXw = np.exp(y * Xw)
                grad_j = lmbd * w[j]
                for i in range(n_samples):
                    grad_j += -y[i] * X[i, j] / (1 + exp_yXw[i])
                if newton_step:
                    w[j] -= grad_j * _newton_step_size(X, exp_yXw, j, lmbd)
                else:
                    w[j] -= grad_j / L[j]
                diff = w[j] - old
                if diff != 0:
                    Xw += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def sparse_cd(X_data, X_indices, X_indptr, y, lmbd, L, n_iter,
                  newton_step):
        n_features = len(X_indptr) - 1
        Xw = np.zeros_like(y)
        exp_yXw = np.zeros_like(y)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                start, end = X_indptr[j:j+2]
                grad_j = lmbd * w[j]
                for ind in range(start, end):
                    i = X_indices[ind]
                    exp_yXw[i] = math.exp(y[i] * Xw[i])
                    grad_j += -y[i] * X_data[ind] / (1 + exp_yXw[i])
                if newton_step:
                    w[j] -= grad_j * _newton_step_size_sparse(
                        X_data, X_indices, X_indptr, exp_yXw, j, lmbd
                    )
                else:
                    w[j] -= grad_j / L[j]

                diff = w[j] - old
                if diff != 0:
                    for ind in range(start, end):
                        Xw[X_indices[ind]] += diff * X_data[ind]
        return w

    def get_result(self):
        return dict(beta=self.w)
