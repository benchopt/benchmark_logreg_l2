from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import _get_cuda_version

cuda_version = _get_cuda_version()

with safe_import_context() as import_ctx:
    from snapml import LogisticRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["pip:snapml"]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.use_gpu = cuda_version is not None

        self.clf = LogisticRegression(
            fit_intercept=False,
            regularizer=self.lmbd,
            penalty="l2",
            tol=1e-12,
            use_gpu=self.use_gpu,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.clf.coef_ = np.zeros(self.X.shape[1])
            return

        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
