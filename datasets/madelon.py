from benchopt import BaseDataset

from libsvmdata import fetch_libsvm
from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):
    name = "madelon"

    install_cmd = 'conda'
    requirements = ['libsvmdata', 'scikit-learn']

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):
        X, y = fetch_libsvm("madelon")
        X_test, y_test = fetch_libsvm("madelon_test")

        if self.scaled:
            # column scaling
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        data = dict(X=X, y=y, X_test=X_test, y_test=y_test)
        return data
