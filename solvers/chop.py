import warnings
import os
from benchopt import BaseSolver, safe_import_context
from torch.utils.data.dataset import TensorDataset

with safe_import_context() as import_ctx:
    import torch
    from torch.utils.data import DataLoader
    import chop


class Solver(BaseSolver):
    name = 'chop'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/chop/archive/master.zip']

    parameters = {
        'solver': ['pgd'],
        'line_search': [False, True],
        'stochastic': [False, True],
        'batch_size': [32, 128, 200, 'full'],
        'normalization': ['none', 'L2', 'Linf', 'sign'],
        'momentum': [0., 0.9],
        'device': ['cuda', 'cpu']
        }

    def skip(self, X, y, lmbd):
        if self.device == 'cuda' and not torch.cuda.is_available():
            return True, "CUDA is not available."

        if not self.stochastic and self.batch_size != 'full':
            return True, "We only run stochastic=False if batch_size=='full'."

        if self.stochastic:
            if self.batch_size == 'full':
                msg = "We do not perform full batch optimization "\
                    "with a stochastic optimizer."
                return True, msg

            if self.line_search:
                msg = "We do not perform line-search with a "\
                    "stochastic optimizer."
                return True, msg

        else:
            if self.normalization != 'none':
                msg = 'Normalizations are not used for full batch '\
                    'optimizers.'
                return True, msg

            if self.momentum != 0.:
                msg = 'Momentum has no impact on full batch optimizers.'
                return True, msg

        return False, None

    def set_objective(self, X, y, lmbd):
        self.lmbd = lmbd

        if self.device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        device = torch.device(self.device)
        self.X = torch.tensor(X).to(device)
        self.y = torch.tensor(y > 0, dtype=torch.float64).to(device)

    def run(self, n_iter):
        X, y, solver = self.X, self.y, self.solver

        n_points, n_features = X.shape

        x0 = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        if n_iter == 0:
            self.beta = x0
            return

        criterion = torch.nn.BCEWithLogitsLoss()

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if self.stochastic:

            # prepare dataset
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size)

            # prepare opt variable
            x = x0.clone().detach().flatten()
            x.requires_grad_(True)

            if solver == 'pgd':
                optimizer = chop.stochastic.PGD([x], lr=.05, momentum=self.momentum,
                                                normalization=self.normalization)

            else:
                raise NotImplementedError

            # Optimization loop
            counter = 0

            alpha = self.lmbd / X.size(0)

            while counter < n_iter:

                for data, target in loader:
                    counter += 1

                    optimizer.zero_grad()
                    pred = data @ x
                    loss = criterion(pred, target)
                    loss += .5 * alpha * (x ** 2).sum()
                    loss.backward()
                    optimizer.step()

            self.beta = x.detach().clone()

        # Full batch optimizers
        else:

            # Set up the problem

            # chop's full batch optimizers require
            # (batch_size, *shape) shape
            x0 = x0.reshape(1, -1)
            @chop.utils.closure
            def logloss(x):

                alpha = self.lmbd / X.size(0)
                out = chop.utils.bmv(X, x)
                loss = criterion(out, y)
                reg = .5 * alpha * (x ** 2).sum()
                return loss + reg

            # Solve the problem
            if solver == 'pgd':
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

    def get_result(self):
        return self.beta.detach().cpu().numpy().flatten()
