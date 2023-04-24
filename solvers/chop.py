import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    # see https://github.com/pytorch/pytorch/issues/78490
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import TensorDataset
    import chop


class Solver(BaseSolver):
    name = 'chop'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/chop/archive/master.zip']

    parameters = {
        'solver': ['pgd'],
        'line_search': [False, True],
        'stochastic': [False, True],
        'batch_size': ['full', 1],
        'momentum': [0., 0.7],
        'device': ['cpu', 'cuda']
    }

    def skip(self, X, y, lmbd):
        if self.device == 'cuda' and not torch.cuda.is_available():
            return True, "CUDA is not available."

        if self.stochastic:
            if self.batch_size == 'full':
                msg = "We do not perform full batch optimization "\
                    "with a stochastic optimizer."
                return True, msg

            if self.line_search:
                msg = "We do not perform line-search with a "\
                    "stochastic optimizer."
                return True, msg

            if self.device == 'cpu':
                msg = "Stochastic optimizers are too slow on cpu."
                return True, msg

        else:  # Full batch methods
            if self.batch_size != 'full':
                return True, "We only run stochastic=False if "\
                    "batch_size=='full'."

            if self.momentum != 0.:
                msg = 'Momentum is not used for full batch optimizers.'
                return True, msg

        return False, None

    def set_objective(self, X, y, lmbd):
        self.lmbd = lmbd

        device = torch.device(self.device)

        self.X = torch.tensor(X).to(device)
        self.y = torch.tensor(y > 0, dtype=torch.float64).to(device)

        _, n_features = X.shape

        self.x0 = torch.zeros(n_features,
                              dtype=self.X.dtype,
                              device=self.X.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def run_stochastic(self, n_iter):
        # prepare dataset
        dataset = TensorDataset(self.X, self.y)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        # prepare opt variable
        x = self.x0.clone().detach().flatten()
        x.requires_grad_(True)

        if self.solver == 'pgd':
            optimizer = chop.stochastic.PGD([x], lr=.05,
                                            momentum=self.momentum)

        else:
            raise NotImplementedError

        # Optimization loop
        counter = 0

        alpha = self.lmbd / self.X.size(0)

        def loglossderiv(p, y):
            z = p * y
            if z > 18:
                return -y * np.exp(-z)
            if z < -18:
                return -y
            return -y / (1. + np.exp(z))

        def optimal_step_size(t):
            """From sklearn, from an idea by Leon Bottou"""
            p = np.sqrt(1. / np.sqrt(alpha))
            eta0 = p / max(1, loglossderiv(-p, 1))
            t0 = 1. / (alpha * eta0)

            return 1. / (alpha * (t0 + t - 1.))

        while counter < n_iter:

            for data, target in loader:
                counter += 1
                optimizer.lr = optimal_step_size(counter)

                optimizer.zero_grad()
                pred = data @ x
                loss = self.criterion(pred, target)
                loss += .5 * alpha * (x ** 2).sum()
                loss.backward()
                optimizer.step()

        self.beta = x.detach().clone()

    def run_full_batch(self, n_iter):
        # Set up the problem

        # chop's full batch optimizers require
        # (batch_size, *shape) shape
        x0 = self.x0.reshape(1, -1)

        @chop.utils.closure
        def logloss(x):

            alpha = self.lmbd / self.X.size(0)
            out = chop.utils.bmv(self.X, x)
            loss = self.criterion(out, self.y)
            reg = .5 * alpha * (x ** 2).sum()
            return loss + reg

        # Solve the problem
        if self.solver == 'pgd':
            if self.line_search:
                step = 'backtracking'
            else:
                # estimate the step using backtracking line search once
                step = None

            result = chop.optim.minimize_pgd(logloss, x0,
                                             prox=lambda x, s=None: x,
                                             step=step,
                                             max_iter=n_iter)

        else:
            raise NotImplementedError

        self.beta = result.x

    def run(self, n_iter):

        if n_iter == 0:
            self.beta = self.x0
            return

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if self.stochastic:
            self.run_stochastic(n_iter)
        else:
            self.run_full_batch(n_iter)

    def get_result(self):
        return self.beta.detach().cpu().numpy().flatten()
