from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import get_cuda_version

cuda_version = get_cuda_version()
if cuda_version is not None:
    cuda_version = cuda_version.split("cuda_", 1)[1][:4]

with safe_import_context() as import_ctx:
    if cuda_version is None:
        raise ImportError("cuml solver needs a nvidia GPU.")

    import cudf
    import numpy as np
    from cuml.linear_model import LogisticRegression


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql",
    ] if cuda_version is not None else []

    parameters = {
        "solver": [
            "qn",
        ],
    }

    support_sparse = False
    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.X = cudf.DataFrame(self.X.astype(np.float32))
        self.y = cudf.Series((self.y > 0).astype(np.float32))

        self.clf = LogisticRegression(
            fit_intercept=False,
            C=1 / self.lmbd,
            penalty="l2",
            tol=1e-15,
            solver=self.solver,
            verbose=0,
        )

    def run(self, n_iter):
        self.clf.solver_model.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.to_numpy().flatten()
