from benchopt import BaseSolver
from benchopt.helpers.requires_gpu import gpu_available
from benchopt.stopping_criterion import SufficientProgressCriterion

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import sparse as jsparse
from scipy import sparse

jax.config.update("jax_enable_x64", True)


@jax.jit
def lbfgs(X, y, lmbd, w0, n_iter):
    n_features = X.shape[1]

    def loss(w):
        y_X_beta = y * (X @ w[:n_features] + w[n_features:].sum())
        return (
            jnp.logaddexp(0, -y_X_beta).sum()
            + lmbd / 2 * w[:n_features] @ w[:n_features]
        )

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(loss)

    def step(_, carry):
        w, state = carry
        value, grad = value_and_grad(w, state=state)
        updates, state = opt.update(
            grad, state, w, value=value, grad=grad, value_fn=loss
        )
        return optax.apply_updates(w, updates), state

    return jax.lax.fori_loop(0, n_iter, step, (w0, opt.init(w0)))[0]


class Solver(BaseSolver):
    """optax L-BFGS with a zoom line search, jitted with JAX on GPU."""
    name = "optax-lbfgs"

    requirements = ["pip::jax[cuda12]", "pip::optax"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def skip(self):
        if not gpu_available(self):
            return "needs a GPU"
        return None

    def set_objective(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X):
            self.X = jsparse.BCOO.from_scipy_sparse(X)
        else:
            self.X = jnp.asarray(X)
        # Keep the dtype of the data, and use it for y and the iterates.
        self.y = jnp.asarray(y, dtype=X.dtype)
        self.lmbd = lmbd
        self.w0 = jnp.zeros(X.shape[1] + fit_intercept, dtype=X.dtype)

        # Run once per problem to absorb its setup costs (jit, CUDA) from the
        # timings, as warm_up only runs once per solver.
        self.run(1)

    def run(self, n_iter):
        self.coef = lbfgs(self.X, self.y, self.lmbd, self.w0, n_iter)
        self.coef.block_until_ready()

    def get_result(self):
        return dict(beta=np.asarray(self.coef))
