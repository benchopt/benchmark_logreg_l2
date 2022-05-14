from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import get_cuda_version


with safe_import_context() as import_ctx:
    from snapml import LogisticRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["pip:snapml"]

    parameters = {"gpu": [False, True]}

    def skip(self, X, y, lmbd):
        if self.gpu and get_cuda_version() is None:
            return True, "snapml[gpu=True] needs a GPU to run"
        return False, None

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.clf = LogisticRegression(
            fit_intercept=False,
            regularizer=self.lmbd,
            penalty="l2",
            tol=1e-12,
            use_gpu=self.gpu,
            dual=self.X.shape[0] >= self.X.shape[1],
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.clf.coef_ = np.zeros(self.X.shape[1])
            return

        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
