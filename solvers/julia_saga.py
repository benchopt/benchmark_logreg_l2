from pathlib import Path
from benchopt.util import safe_import_context

from benchopt.utils.julia_helpers import JuliaSolver
from benchopt.utils.julia_helpers import get_jl_interpreter
from benchopt.utils.julia_helpers import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix('.jl'))


class Solver(JuliaSolver):

    # Config of the solver
    name = 'Julia-SAGA'
    stop_strategy = 'iteration'

    # Julia package dependencies
    julia_requirements = ['StatsBase']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        jl = get_jl_interpreter()
        self.solve_logreg_l2 = jl.include(JULIA_SOLVER_FILE)

    def run(self, n_iter):
        self.beta = self.solve_logreg_l2(self.X, self.y, self.lmbd, n_iter)

    def get_result(self):
        return self.beta.ravel()
