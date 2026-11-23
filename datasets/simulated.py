from benchopt import BaseDataset


import numpy as np

from benchmark_utils.lmbd_cv import get_lmbd_cv


class Dataset(BaseDataset):

    name = "Simulated"

    # For the selection of lmbd_cv.
    requirements = ["scikit-learn"]

    parameters = {
        'n_samples, n_features': [
            (200, 500),
            (1000, 10),
            (100_000, 400),
        ],
        'scaled': [True, False]
    }

    def prepare(self):
        # Nothing to download, and the data depends on the repetition, which
        # is not known when preparing.
        pass

    def get_data(self):
        seed = self.get_seed(use_dataset=True, use_repetition=True)
        rng = np.random.RandomState(seed)
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

        data = dict(
            X=X, y=y, X_test=X_test, y_test=y_test,
            lmbd_cv=get_lmbd_cv(self, X, y, seed=seed)
        )

        return data
