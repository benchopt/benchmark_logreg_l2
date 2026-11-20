from benchopt import BaseSolver
from benchopt.helpers.requires_gpu import gpu_available
from benchopt.stopping_criterion import SufficientProgressCriterion

import numpy as np
from scipy import sparse

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusparse
from cuml.linear_model import LogisticRegression


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = ["pip::cuml"]

    parameters = {
        "solver": [
            "qn",
        ],
    }
    parameter_template = "{solver}"

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        if not gpu_available(self):
            return "cuml solver needs a nvidia GPU."
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.lmbd = lmbd

        if sparse.issparse(X):
            if sparse.isspmatrix_csc(X):
                self.X = cusparse.csc_matrix(X)
            elif sparse.isspmatrix_csr(X):
                self.X = cusparse.csr_matrix(X)
            else:
                raise ValueError("Non suported sparse format")
        else:
            self.X = cudf.DataFrame(X.astype(np.float32))

        self.y = cudf.Series((y > 0).astype(np.float32))

        self.clf = LogisticRegression(
            fit_intercept=fit_intercept,
            C=1 / self.lmbd,
            penalty="l2",
            tol=1e-15,
            solver="qn",
            verbose=0,
        )

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        if isinstance(self.clf.coef_, cp.ndarray):
            coef = self.clf.coef_.get().flatten()
            if self.clf.fit_intercept:
                coef = np.r_[coef, self.clf.intercept_.get()]
        else:
            coef = self.clf.coef_.to_numpy().flatten()
            if self.clf.fit_intercept:
                coef = np.r_[coef, self.clf.intercept_.to_numpy()]

        return dict(beta=coef.astype(np.float64))
