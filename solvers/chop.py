import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import TensorDataset
    import chop


class Solver(BaseSolver):
    name = 'chop'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/chop/archive/master.zip',
                    'pip:scikit-learn']

    parameters = {
        'solver': ['pgd'],
        'line_search': [True, False],
        'stochastic': [False, True],
        'batch_size': ['full', 1],
        'momentum': [0., 0.9],
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

        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

        _, n_features = X.shape

        self.x0 = torch.zeros(n_features,
                              dtype=self.X.dtype,
                              device=self.X.device)

        # prepare loader for stochastic methods
        if self.stochastic:
            dataset = TensorDataset(self.X, self.y)
            self.loader = DataLoader(dataset, batch_size=self.batch_size)

        def logloss(x, data=self.X, target=self.y):
            y_X_x = target * (data @ x.flatten())
            l2 = 0.5 * x.pow(2).sum()
            loss = torch.log1p(torch.exp(-y_X_x)).sum() + self.lmbd * l2
            return loss

        self.objective = logloss

    def run_stochastic(self, n_iter):

        # prepare opt variable
        x = self.x0.clone().detach()
        x.requires_grad_(True)

        if self.solver == 'pgd':
            optimizer = chop.stochastic.PGD([x], lr=.05,
                                            momentum=self.momentum)

        else:
            raise NotImplementedError

        # Optimization loop

        alpha = self.lmbd / self.X.size(0)

        def loglossderiv(p, y):
            z = p * y
            if z > 18:
                return -y * np.exp(-z)
            if z < -18:
                return -y
            return -y / (1. + np.exp(z))

        def initial_step_size():
            p = np.sqrt(1. / np.sqrt(alpha))
            eta0 = p / max(1, loglossderiv(-p, 1))
            t0 = 1. / (alpha * eta0)
            return t0

        t0 = initial_step_size()

        def optimal_step_size(t):
            """From sklearn, from an idea by Leon Bottou"""
            return 1. / (alpha * (t0 + t - 1.))

        counter = 0
        stop = False
        while not stop:
            for data, target in self.loader:
                counter += 1
                if counter == n_iter:
                    stop = True
                    break
                optimizer.lr = optimal_step_size(counter)
                optimizer.zero_grad()
                loss = self.objective(x, data=data, target=target)
                loss.backward()
                optimizer.step()

        self.beta = x.detach().clone()

    def run_full_batch(self, n_iter):
        # Set up the problem
        @chop.utils.closure
        def objective(x):
            return self.objective(x, data=self.X, target=self.y)

        # chop's full batch optimizers require
        # (batch_size, *shape) shape
        x0 = self.x0.reshape(1, -1)

        # Solve the problem
        if self.solver == 'pgd':
            if self.line_search:
                step = 'backtracking'
            else:
                # estimate the step using backtracking line search once
                step = None

            result = chop.optim.minimize_pgd(objective, x0,
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
