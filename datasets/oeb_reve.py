from benchopt import BaseDataset
from benchopt.config import get_data_path

import numpy as np
from sklearn.preprocessing import StandardScaler

from benchmark_utils.lmbd_cv import get_lmbd_cv
from benchmark_utils.oeb_reve import encode_reve


class Dataset(BaseDataset):
    """Linear probing of the REVE EEG foundation model on Open EEG Bench.

    The windows of a binary Open EEG Bench task are encoded with the frozen
    REVE-base encoder, as in Open EEG Bench's frozen linear probe: same
    windows, subject split, normalization and features, i.e. all the encoder
    tokens flattened, then standardized. The logistic regression is fit on
    the train and validation subjects, the test subjects give the test
    metrics.
    """
    name = "OpenEEGBench-REVE"

    requirements = ["pip::open-eeg-bench", "pip::torch", "scikit-learn"]

    references = [
        "P. Guetschel et al., \"Toward OpenEEG-Bench: A Live "
        "Community-Driven Benchmark for EEG Foundation Models\", EUSIPCO 2026",
        "Y. El Ouahidi et al., \"REVE: A Foundation Model for EEG - Adapting "
        "to Any Setup with Large-Scale Pretraining on 25,000 Subjects\", "
        "NeurIPS 2025",
    ]

    parameters = {
        # Binary tasks: mental arithmetic vs rest, and depression detection.
        'task': ['arithmetic_zyma2019', 'mdd_mumtaz2016'],
        # Precision of the features given to the solvers, e.g. to compare the
        # GPU solvers in float32 with dtype=float32.
        'dtype': ['float64'],
    }
    # Too many features for the solvers forming a dense hessian.
    large = True
    # The smallest task, as the tests encode it with REVE on CPU.
    test_parameters = {'task': ['arithmetic_zyma2019']}

    def _encode(self):
        """Encode the task windows with the frozen REVE encoder, once."""
        path = get_data_path("open_eeg_bench") / f"{self.task}_reve.npz"
        if path.exists():
            return path

        features = encode_reve(self.task)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **features)
        return path

    def get_data(self):
        # benchopt prepare runs get_data: it encodes the windows once per
        # task, and computes its lmbd_cv.
        data = np.load(self._encode())
        X, X_test = data["X_train"], data["X_test"]
        X, X_test = X.astype(np.float64), X_test.astype(np.float64)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
        # Selected on the float64 features, and stored under the float64 name,
        # so that both dtypes solve the same problem.
        float64 = self.get_instance(task=self.task, dtype="float64")
        lmbd_cv = get_lmbd_cv(float64, X, data["y_train"])

        return dict(
            X=X.astype(self.dtype), y=data["y_train"],
            X_test=X_test.astype(self.dtype), y_test=data["y_test"],
            lmbd_cv=lmbd_cv,
        )
