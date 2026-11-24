from benchopt import BaseDataset

from libsvmdata import fetch_libsvm
from sklearn.preprocessing import StandardScaler

from benchmark_utils.lmbd_cv import get_lmbd_cv


class Dataset(BaseDataset):
    name = "rcv1"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip::libsvmdata', "scikit-learn"]

    # Too many features for the solvers forming a dense hessian.
    large = True

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):

        X, y = fetch_libsvm('rcv1.binary', min_nnz=0)
        X_test, y_test = fetch_libsvm('rcv1.binary_test', min_nnz=0)

        if self.scaled:
            # column scaling - sparse dataset so no mean
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        data = dict(
            X=X, y=y, X_test=X_test, y_test=y_test,
            lmbd_cv=get_lmbd_cv(self, X, y)
        )

        return data
