from benchopt import BaseSolver
from benchopt.utils.sys_info import get_cuda_version
from benchopt.stopping_criterion import SufficientProgressCriterion

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusparse
from cuml.linear_model import LogisticRegression

<<<<<<< HEAD
from benchopt import BaseSolver, safe_import_context
from benchopt.helpers.requires_gpu import requires_gpu

cuda_version = None
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    cuda_version = requires_gpu()

    if cuda_version is not None:
=======



import cudf
import numpy as np
from cuml.linear_model import LogisticRegression

cuda_version = get_cuda_version()
if cuda_version is not None:
    cuda_version = cuda_version.split("cuda_", 1)[1][:4]

if cuda_version is None:
    raise ImportError("cuml solver needs a nvidia GPU.")
>>>>>>> main


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "pip::cuml",
    ]

    parameters = {
        "solver": [
            "qn",
        ],
    }

    support_sparse = False
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        if sparse.issparse(X):
            if sparse.isspmatrix_csc(X):
                self.X = cusparse.csc_matrix(X)
            elif sparse.isspmatrix_csr(X):
                self.X = cusparse.csr_matrix(X)
            else:
                raise ValueError("Non suported sparse format")
        else:
            self.X = cudf.DataFrame(self.X.astype(np.float32))

        self.y = cudf.Series((self.y > 0).astype(np.float32))

        self.clf = LogisticRegression(
            fit_intercept=fit_intercept,
            C=1 / self.lmbd,
            penalty="l2",
            tol=1e-15,
            solver="qn",
            verbose=0,
        )

    def run(self, n_iter):
        self.clf.solver_model.max_iter = n_iter
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
