from benchopt import BaseDataset


import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (200, 500),
            (1000, 10),
            (100_000, 400),
        ],
        'scaled': [True, False]
    }

    def __init__(self, n_samples=10, n_features=50, scaled=True,
                 random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        beta = rng.randn(self.n_features)

        X = rng.randn(self.n_samples, self.n_features)
        y = np.sign(X @ beta)

        X_test = rng.randn(self.n_samples, self.n_features)
        y_test = np.sign(X_test @ beta)

        if self.scaled:
            # column scaling
            mu, sigma = X.mean(axis=0), X.std(axis=0)
            X -= mu
            X /= sigma
            X_test -= mu
            X_test /= sigma

        data = dict(X=X, y=y, X_test=X_test, y_test=y_test)

        return data
