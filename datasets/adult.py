from benchopt import BaseDataset


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer


class Dataset(BaseDataset):
    name = "adult"

    install_cmd = 'conda'
    requirements = ["scikit-learn"]

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):

        data = fetch_openml(data_id=1590, as_frame=True)
        X = pd.get_dummies(data.data).values
        label_encoder = LabelBinarizer(neg_label=-1)
        y = label_encoder.fit_transform(data.target)[:, 0]

        if self.scaled:
            # column scaling - most features are sparse so no mean
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        data = dict(X=X, y=y)

        return data
