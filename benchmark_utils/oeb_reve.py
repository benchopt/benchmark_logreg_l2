import numpy as np
import torch
from open_eeg_bench.default_configs import backbones, datasets
from open_eeg_bench.head import FlattenHead
from torch.utils.data import ConcatDataset, DataLoader


def encode_reve(task):
    """Frozen REVE features of an Open EEG Bench task, as in its linear probe.

    Returns the flattened encoder tokens and the labels in {-1, 1}, for the
    train and validation subjects (train) and for the test subjects (test).
    """
    dataset = getattr(datasets, task)()
    backbone = backbones.reve()
    _, train, val, test, info = dataset.setup(
        normalization=backbone.normalization
    )
    model = backbone.build(
        n_chans=info["n_chans"], n_times=info["n_times"],
        n_outputs=info["n_outputs"], sfreq=info["sfreq"],
        chs_info=info["chs_info"],
    )
    FlattenHead().apply(model, info["n_outputs"], backbone.head_module_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    features = {}
    for split, windows in [("train", ConcatDataset([train, val])),
                           ("test", test)]:
        X, y = [], []
        with torch.inference_mode():
            for X_batch, y_batch, *_ in DataLoader(windows, batch_size=64):
                X.append(model(X_batch.to(device)).cpu().numpy())
                y.append(np.asarray(y_batch))
        features[f"X_{split}"] = np.concatenate(X)
        features[f"y_{split}"] = 2. * np.concatenate(y) - 1
    return features
