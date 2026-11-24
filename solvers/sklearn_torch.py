import os
import warnings

from benchopt import BaseSolver
from benchopt.helpers.requires_gpu import gpu_available
from benchopt.stopping_criterion import SufficientProgressCriterion

import torch
from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class Solver(BaseSolver):
    """sklearn's lbfgs LogisticRegression on GPU through the array API."""
    name = "sklearn-torch"

    requirements = ["pip::torch", "scikit-learn"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        # scipy reads this variable at import, so it must be set in the
        # environment of the run, before benchopt imports scipy.
        if os.environ.get("SCIPY_ARRAY_API") != "1":
            return "needs SCIPY_ARRAY_API=1 in the environment"
        if not gpu_available(self):
            return "needs a GPU"
        if getattr(self.dataset_, "sparse", False):
            return "the array API does not support sparse data"
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        # Keep the dtype of the data, and use it for y.
        self.X = torch.as_tensor(X, device="cuda")
        self.y = torch.as_tensor(y, dtype=self.X.dtype, device="cuda")

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf = LogisticRegression(
            C=1 / lmbd, fit_intercept=fit_intercept, tol=1e-15
        )

        # Run once per problem to absorb its setup costs (jit, CUDA) from the
        # timings, as warm_up only runs once per solver.
        self.run(1)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        with config_context(array_api_dispatch=True):
            self.clf.fit(self.X, self.y)
        # Count the asynchronous GPU work in the timing.
        torch.cuda.synchronize()

    def get_result(self):
        coef = self.clf.coef_.flatten()
        if self.clf.fit_intercept:
            coef = torch.cat([coef, self.clf.intercept_])
        return dict(beta=coef.cpu().numpy())
