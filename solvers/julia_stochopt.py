from pathlib import Path
from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = Path(__file__).with_suffix('.jl')


class Solver(JuliaSolver):

    name = 'julia-stochopt'
    references = [
        'Gazagnadou, Gower, and Salmon,'
        ' "Optimal mini-batch and step sizes for SAGA." '
        'International conference on machine learning. PMLR, 2019.'
        'Sebbouh, O., et al.  Towards closing the gap between the theory and'
        'practice of SVRG. Advances in Neural Information Processing Systems, 2019'
        'https://github.com/gowerrobert/StochOpt.jl'
    ]

    # List of dependencies can be found on the package github
    julia_requirements = [
        'StochOpt::https://github.com/tommoral/StochOpt.jl#FIX_proper_module_install',  # noqa: E501
    ]

    parameters = {
        'method': [
            "SVRG",
            "BFGS",
            "AMgauss",  # A variant of SVRG that uses a embed Hessian matrix
            "SAGA_nice",  # SAGA sampling with closed-form optimal mini-batch
            "SVRG_bubeck",  
            "Leap_SVRG",  # SVRG without outer loop but a coin tossing at each
                          # iteration to decide whether the reference is
                          # updated or not
            "L_SVRG_D"  # Loopless-SVRG-Decreasing without outer loop but a
                        # coin tossing at each iteration to decide whether te
                        # reference is updated (with probability p) or not
        ],
        'batch_size': [128]
    }

    def set_objective(self, X, y, lmbd):

        self.X, self.y, self.lmbd = X, y, lmbd
        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_logreg = jl.solve_logreg

        # run iteration once to cache the JIT computation
        self.solve_logreg(
            self.X.T, self.y, self.lmbd, 2,
            self.method, self.batch_size
        )

    def run(self, n_iter):
        self.coef_ = self.solve_logreg(
            self.X.T, self.y, self.lmbd, n_iter,
            self.method, self.batch_size
        )

    def get_result(self):
        return self.coef_.ravel()
