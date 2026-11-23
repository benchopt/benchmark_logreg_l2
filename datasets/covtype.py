from benchopt import BaseDataset


from libsvmdata import fetch_libsvm
from sklearn.preprocessing import StandardScaler

from benchmark_utils.lmbd_cv import get_lmbd_cv


class Dataset(BaseDataset):
    name = "covtype_binary"

    install_cmd = 'conda'
    requirements = ['pip::libsvmdata', 'scikit-learn']

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):
        X, y = fetch_libsvm("covtype.binary")

        # Data is sparse, convert to dense array
        X = X.toarray()

        # Labels are {1,2}, convert to {-1, 1}
        y = (2 * y - 3).astype(int)

        # This dataset contains a mixture of numeric columns and
        # one-hot-encoded categorical columns, scale it to avoid very large
        # condition number.
        if self.scaled:
            # column scaling
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        return dict(
            X=X, y=y,
            lmbd_cv=get_lmbd_cv(self, X, y)
        )
