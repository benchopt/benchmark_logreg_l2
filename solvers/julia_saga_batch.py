from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = "solvers/julia_stochopt.jl"


class Solver(JuliaSolver):

    name = 'julia-saga-batch'
    references = [
        'Gazagnadou, Gower, and Salmon,'
        ' "Optimal mini-batch and step sizes for SAGA." '
        'International conference on machine learning. PMLR, 2019.'
        'https://github.com/gowerrobert/StochOpt.jl'
    ]

    # List of dependencies can be found on the package github
    julia_requirements = [
        'StochOpt::https://github.com/tommoral/StochOpt.jl#FIX_proper_module_install',  # noqa: E501
    ]

    def set_objective(self, X, y, lmbd):

        self.X, self.y, self.lmbd = X, y, lmbd
        jl = get_jl_interpreter()
        jl.include(JULIA_SOLVER_FILE)
        self.solve_logreg_saga_batch = jl.solve_logreg_saga_batch

    def run(self, n_iter):
        self.beta = self.solve_logreg_saga_batch(
            self.X.T, self.y, self.lmbd, n_iter
        )[1:]

    def get_result(self):
        return self.beta.ravel()
