from pathlib import Path
from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix('.jl'))


class Solver(JuliaSolver):

    name = 'julia-svrg'
    references = [
        'Gazagnadou, Nidham, Robert Gower, and Joseph Salmon,'
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
        self.solve_logreg_l2 = jl.include(JULIA_SOLVER_FILE)

    def run(self, n_iter):
        self.beta = self.solve_logreg_l2(
            self.X.transpose(), self.y, self.lmbd, n_iter)[1:]

    def get_result(self):
        return self.beta.ravel()
