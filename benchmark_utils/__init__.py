import json
import warnings

import numpy as np
from benchopt.config import get_data_path
from sklearn.linear_model import LogisticRegressionCV


def get_lmbd_cv(dataset, X, y):
    """Return the lmbd selected by cross-validation for each fit_intercept.

    The values are computed once, when preparing the dataset, and stored in a
    json file in the data directory, keyed by str(fit_intercept).
    """
    path = get_data_path("lmbd_cv") / f"{dataset}.json"
    if not path.exists():
        lmbd_cv = {}
        for fit_intercept in [False, True]:
            warnings.filterwarnings("ignore", category=FutureWarning)
            clf = LogisticRegressionCV(
                Cs=np.logspace(-8, 8, 33), solver="newton-cg",
                scoring="neg_log_loss", fit_intercept=fit_intercept, n_jobs=-1
            ).fit(X, y)
            # sklearn minimizes C * sum(loss) + ||w||^2 / 2, so lmbd = 1 / C.
            lmbd_cv[str(fit_intercept)] = 1 / clf.C_[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lmbd_cv, indent=2))
    return json.loads(path.read_text())
