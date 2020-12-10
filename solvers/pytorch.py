import numpy as np
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import torch
    import torch.utils.data


class Solver(BaseSolver):
    name = 'Pytorch-SGD'

    install_cmd = 'conda'
    requirements = [
        'pytorch-cpu'
    ]

    def set_objective(self, X, y, lmbd):

        self.X, self.y, self.lmbd = X, y, lmbd

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float64),
            (1 + torch.tensor(y, dtype=torch.float64)) / 2
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=X.shape[0], shuffle=False
        )

        self.model = torch.nn.Linear(
            in_features=X.shape[1], out_features=1, bias=False
        ).to(torch.float64)
        self.w = self.model.weight.view(-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss()

    def run(self, n_iter):

        L = np.linalg.norm(self.X, ord=2) ** 2 / 4 + self.lmbd

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1 / L)

        for epoch in range(n_iter):
            for X_batch, y_batch in self.train_loader:

                self.optimizer.zero_grad()
                outputs = self.model(X_batch).view(-1)
                outputs = self.sigmoid(outputs)
                loss = self.loss(outputs, y_batch)
                loss += self.lmbd / 2 * self.w.dot(self.w)
                loss.backward()
                self.optimizer.step()

    def get_result(self):
        return self.model.weight.detach().cpu().numpy().flatten()
