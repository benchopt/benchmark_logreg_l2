import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning

    from skglm.penalties import L2
    from skglm.solvers import LBFGS
    from skglm.datafits import Logistic

    from skglm.utils.jit_compilation import compiled_clone


class Solver(BaseSolver):
    name = 'skglm'
    stopping_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/scikit-learn-contrib/skglm.git@main'
    ]

    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = X.shape[0]

        self.datafit = compiled_clone(Logistic())
        self.penalty = compiled_clone(L2(lmbd / n_samples))

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.solver = LBFGS(tol=1e-20)

        # cache Numba compilation
        self.run(3)

    def run(self, n_iter):
        self.solver.max_iter = n_iter

        self.coef_ = self.solver.solve(
            self.X, self.y, self.datafit, self.penalty)[0]

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.coef_.flatten()
