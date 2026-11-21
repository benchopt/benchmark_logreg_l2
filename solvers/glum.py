import warnings

from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from scipy import sparse
from glum import GeneralizedLinearRegressor


class Solver(BaseSolver):
    name = "glum"

    requirements = ["pip::glum"]

    parameters = {
        "solver": ["irls-cd", "irls-ls", "lbfgs"],
    }
    parameter_template = "{solver}"

    references = [
        "M. Kreutzer et al., \"glum: Fast GLMs in Python\", "
        "https://github.com/Quantco/glum"
    ]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        n_features = self.objective_.get_objective()["X"].shape[1]
        if self.solver.startswith("irls") and n_features > 10_000:
            return "irls forms a dense n_features x n_features hessian"
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X = X.tocsc() if sparse.issparse(X) else X
        self.y = (y > 0).astype(np.float64)
        self.n_coeffs = X.shape[1] + fit_intercept

        warnings.filterwarnings('ignore', category=UserWarning)
        # glum averages the deviance over samples, so alpha = lmbd / n.
        self.clf = GeneralizedLinearRegressor(
            family="binomial", alpha=lmbd / X.shape[0], l1_ratio=0,
            fit_intercept=fit_intercept, solver=self.solver,
            gradient_tol=1e-15, step_size_tol=1e-15
        )

        # Run once per problem to absorb its setup costs (jit, CUDA) from the
        # timings, as warm_up only runs once per solver.
        self.run(1)

    def run(self, n_iter):
        self.coef = np.zeros(self.n_coeffs)
        if n_iter:
            self.clf.max_iter = n_iter
            self.clf.fit(self.X, self.y)
            self.coef = self.clf.coef_
            if self.clf.fit_intercept:
                self.coef = np.r_[self.coef, self.clf.intercept_]

    def get_result(self):
        return dict(beta=self.coef)
