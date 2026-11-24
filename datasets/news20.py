from benchopt import BaseDataset

from libsvmdata import fetch_libsvm
from sklearn.preprocessing import StandardScaler

from benchmark_utils.lmbd_cv import get_lmbd_cv


class Dataset(BaseDataset):
    name = "news20"
    is_sparse = True

    install_cmd = "conda"
    requirements = ["pip::libsvmdata", "scikit-learn"]

    # Too many features for the solvers forming a dense hessian.
    large = True
    # Sparse features, for the solvers that only support dense ones.
    sparse = True

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):

        X, y = fetch_libsvm("news20.binary")

        if self.scaled:
            # column scaling
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        return dict(
            X=X, y=y,
            lmbd_cv=get_lmbd_cv(self, X, y)
        )
