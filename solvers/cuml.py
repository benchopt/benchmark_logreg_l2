from benchopt import BaseSolver, safe_import_context
from benchopt.helpers.requires_gpu import requires_gpu
from benchopt.stopping_criterion import SufficientProgressCriterion

cuda_version = None
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    cuda_version = requires_gpu()

    if cuda_version is not None:
        import cudf
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        from cuml.linear_model import LogisticRegression


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        "nvidia::cudatoolkit",
        "cupy"
    ] if cuda_version is not None else []

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

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

        return coef.astype(np.float64)
