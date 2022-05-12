from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import _get_cuda_version

cuda_version = _get_cuda_version()
if cuda_version is not None:
    cuda_version = cuda_version.split("cuda_", 1)[1][:4]

with safe_import_context() as import_ctx:
    import pynvml
    try:
        pynvml.nvmlInit()
    except pynvml.NVMError:
        raise ImportError("Pynvml was enable to locate NVML lib.")
    n_gpus = pynvml.nvmlDeviceGetCount()
    if n_gpus < 1:
        raise ImportError("Need a GPU to run cuml solver.")

    import cudf
    import numpy as np
    from cuml.linear_model import LogisticRegression


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        "nvidia::cudatoolkit",
        "dask-sql", "pynvml"
    ] if cuda_version is not None else []

    parameter_template = "{solver}"

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
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
        coef = self.clf.coef_.to_numpy().flatten()
        if self.clf.fit_intercept:
            coef = np.r_[coef, self.clf.intercept_.to_numpy()]
        return coef
