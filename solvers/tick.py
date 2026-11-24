from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from scipy import sparse
from tick.linear_model import ModelLogReg
from tick.prox import ProxL2Sq, ProxZero
from tick.solver import SAGA, SDCA, SVRG


class Solver(BaseSolver):
    name = 'tick'

    install_cmd = 'conda'
    # tick imports numpydoc and six at runtime without declaring them.
    requirements = ['numpydoc', 'six', 'pip::tick>=0.8.0.2']

    parameters = {
        'solver': ['svrg', 'saga', 'sdca'],
    }
    parameter_template = "{solver}"

    # A patience of 10 points lets the stochastic solvers pass their first
    # iterations, where the loss can be above its initial value.
    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=10, strategy='iteration'
    )

    references = [
        "E. Bacry, M. Bompaire, S. Gaïffas and S. V. Poulsen, "
        "\"tick: a Python library for statistical learning, with a "
        "particular emphasis on time-dependent modeling\", "
        "arXiv:1707.03003 (2017)"
    ]

    def skip(self):
        # SDCA penalizes all the coefficients, including the intercept.
        if self.solver == 'sdca' and self.objective_.fit_intercept:
            return "sdca penalizes the intercept"
        return None

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
        if self.solver == 'sdca':
            prox = ProxZero()
            self.optimizer = SDCA(
                l_l2sq=lmbd / n_samples, tol=0, verbose=False, seed=42
            )
        else:
            prox = ProxL2Sq(strength=lmbd / n_samples, range=(0, n_features))
            klass = dict(svrg=SVRG, saga=SAGA)[self.solver]
            self.optimizer = klass(
                step=1 / self.model.get_lip_max(), tol=0, verbose=False,
                seed=42,
            )
        self.optimizer.set_model(self.model).set_prox(prox)

    def run(self, n_iter):
        self.coef = np.zeros(self.model.n_coeffs)
        if n_iter:
            self.optimizer.max_iter = n_iter
            self.coef = self.optimizer.solve()

    def get_result(self):
        return dict(beta=self.coef)
