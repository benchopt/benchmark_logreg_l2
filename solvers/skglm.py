from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from scipy import sparse
from skglm.datafits import Logistic
from skglm.penalties import L1_plus_L2, L2
from skglm.solvers import LBFGS, AndersonCD


class Solver(BaseSolver):
    name = "skglm"

    requirements = ["pip::skglm"]

    parameters = {
        "solver": ["anderson-cd", "lbfgs"],
    }
    parameter_template = "{solver}"

    references = [
        "Q. Bertrand, Q. Klopfenstein, P.-A. Bannier, G. Gidel and "
        "M. Massias, \"Beyond L1: Faster and Better Sparse Models with "
        "skglm\", NeurIPS 2022"
    ]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        if self.solver == "lbfgs" and self.objective_.fit_intercept:
            return "lbfgs does not fit an intercept"
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X = X.tocsc() if sparse.issparse(X) else X
        self.y = y
        self.n_coeffs = X.shape[1] + fit_intercept

        # skglm averages the loss over samples, so the penalty is scaled by
        # 1 / n_samples. CD needs a prox, which L2 lacks, so it uses
        # L1_plus_L2 with l1_ratio=0, the same penalty.
        self.datafit = Logistic()
        alpha = lmbd / X.shape[0]
        if self.solver == "lbfgs":
            self.penalty = L2(alpha)
            self.optimizer = LBFGS(tol=1e-12)
        else:
            self.penalty = L1_plus_L2(alpha, l1_ratio=0.)
            # Bound the inner epochs so that one run stays short.
            self.optimizer = AndersonCD(
                tol=1e-12, max_epochs=1000, fit_intercept=fit_intercept
            )

        # Run once per problem to absorb its setup costs (jit, CUDA) from the
        # timings, as warm_up only runs once per solver.
        self.run(1)

    def run(self, n_iter):
        self.coef = np.zeros(self.n_coeffs)
        if n_iter:
            self.optimizer.max_iter = n_iter
            self.coef = self.optimizer.solve(
                self.X, self.y, self.datafit, self.penalty
            )[0]

    def get_result(self):
        return dict(beta=self.coef)
