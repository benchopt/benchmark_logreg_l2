from benchopt import BaseSolver

import numpy as np
from scipy import sparse
from tick.linear_model import ModelLogReg
from tick.prox import ProxL2Sq
from tick.solver import SVRG


class Solver(BaseSolver):
    name = 'svrg-tick'

    install_cmd = 'conda'
    # tick imports numpydoc and six at runtime without declaring them.
    requirements = ['numpydoc', 'six', 'pip::tick>=0.8.0.2']

    references = [
        "E. Bacry, M. Bompaire, S. Gaïffas and S. V. Poulsen, "
        "\"tick: a Python library for statistical learning, with a "
        "particular emphasis on time-dependent modeling\", "
        "arXiv:1707.03003 (2017)"
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        n_samples, n_features = X.shape
        if sparse.issparse(X):
            X = sparse.csr_matrix(X, dtype=np.float64)
        else:
            X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        # tick averages the loss over samples, so the penalty is scaled by
        # 1 / n_samples. The intercept, last in the coefficients, is not
        # penalized.
        self.model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
        prox = ProxL2Sq(strength=lmbd / n_samples, range=(0, n_features))
        self.solver = SVRG(
            step=1 / self.model.get_lip_max(), tol=0, verbose=False,
            seed=42,
        )
        self.solver.set_model(self.model).set_prox(prox)

    def run(self, n_iter):
        self.coef = np.zeros(self.model.n_coeffs)
        if n_iter:
            self.solver.max_iter = n_iter
            self.coef = self.solver.solve(self.coef)

    def get_result(self):
        return dict(beta=self.coef)
