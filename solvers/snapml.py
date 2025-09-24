from benchopt import BaseSolver
from benchopt.utils.sys_info import get_cuda_version
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from snapml import LogisticRegression


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["pip::snapml"]

    parameters = {"gpu": [False, True]}

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=10, strategy='iteration'
    )

    def skip(self, X, y, lmbd, fit_intercept):
        if self.gpu and get_cuda_version() is None:
            return True, "snapml[gpu=True] needs a GPU to run"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.clf = LogisticRegression(
            fit_intercept=fit_intercept,
            regularizer=self.lmbd,
            penalty="l2",
            tol=1e-12,
            use_gpu=self.gpu,
            dual=False,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.clf.coef_ = np.zeros(self.X.shape[1])
            self.clf.intercept_ = np.zeros(1)
            return

        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        coef = self.clf.coef_.flatten()
        if self.clf.fit_intercept:
            coef = np.r_[coef, self.clf.intercept_]
        return dict(beta=coef)
