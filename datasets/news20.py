from benchopt import BaseDataset


from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "news20"
    is_sparse = True

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm("news20.binary")

        data = dict(X=self.X, y=self.y)

        return data
