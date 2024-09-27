from pathlib import Path
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()
    from scipy import sparse


# File containing the function to be called from julia
JULIA_SOLVER_FILE = Path(__file__).with_suffix('.jl')


class Solver(JuliaSolver):

    name = 'julia-stochopt'
    references = [
        '[1] Gazagnadou, Gower, and Salmon,'
        ' "Optimal mini-batch and step sizes for SAGA." '
        'International conference on machine learning. PMLR, 2019.'
        '[2] Sebbouh et al.  Towards closing the gap between the theory and'
        'practice of SVRG. Advances in Neural Information Processing Systems, 2019'
        'https://github.com/gowerrobert/StochOpt.jl'
    ]

    # List of dependencies can be found on the package github
    julia_requirements = [
        'StochOpt::https://github.com/tommoral/StochOpt.jl#FIX_linear_algebra_deps',  # noqa: E501
        'PyCall',
    ]

    parameters = {
        'method': [
            "SVRG",
            "Free_SVRG", # SVRG with inner loop not averaged at any point [2]
            "BFGS",
            "AMgauss",  # A variant of SVRG that uses a embed Hessian matrix
            "SAGA_nice",  # SAGA sampling with closed-form optimal mini-batch [1]
            "SVRG_bubeck",
            "Leap_SVRG",  # SVRG without outer loop [2]
            "L_SVRG_D",  # Loopless-SVRG-Decreasing [2]
            "Free_SVRG_2n",
            "Free_SVRG_lmax",
            "Free_SVRG_mstar",
        ],
        'batch_size': [
            1,  # to allow calculation of theoretical step size to some methods
            128
        ],
        'numinneriters': [
            1,
            -1
        ]
    }

    stopping_criterion = SufficientProgressCriterion(
        strategy='iteration', patience=10
    )

    def set_objective(self, X, y, lmbd):

        self.X, self.y, self.lmbd = X, y, lmbd

        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_logreg = jl.solve_logreg

        # transform scipy sparse array to julia sparse array
        if isinstance(X, sparse.csc_array):
            scipyCSC_to_julia = jl.pyfunctionret(
                jl.Main.scipyCSC_to_julia, jl.Any, jl.PyObject
            )
            self.X = scipyCSC_to_julia(X)

        # set optimal inner_loop_iter equals n_samples -- to check table 1 [2]
        if self.method == 'Free_SVRG' and self.numinneriters != -1:            
            self.numinneriters = X.shape[0]

        # run iteration once to cache the JIT computation
        self.solve_logreg(
            self.X.T, self.y, self.lmbd, 2,
            self.method, self.batch_size, self.numinneriters
        )

    def run(self, n_iter):
        self.coef_ = self.solve_logreg(
            self.X.T, self.y, self.lmbd, n_iter,
            self.method, self.batch_size, self.numinneriters
        )

    def get_result(self):
        return self.coef_.ravel()
