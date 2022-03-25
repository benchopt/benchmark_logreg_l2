from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from snapml import LogisticRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["libgcc-ng", "pip:snapml"]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        self.clf = LogisticRegression(
            fit_intercept=False,
            regularizer=self.lmbd,
            penalty="l2",
            tol=1e-12,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.clf.coef_ = np.zeros(self.X.shape[1])
            return

        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
