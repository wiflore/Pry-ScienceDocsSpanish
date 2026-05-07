"""Creación de splits train/val/test estratificados por etiqueta."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset, DatasetDict


def create_splits(
    dataset: Dataset,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    label_column: str = "imrad_label",
    save_dir=None,
) -> DatasetDict:
    """Divide el dataset en train/validation/test con estratificación."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        if save_dir.exists():
            return DatasetDict.load_from_disk(str(save_dir))

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio debe ser < 1.0")

    labels = np.array(dataset[label_column])
    indices = np.arange(len(dataset))

    # Primera partición: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    # Segunda partición: val vs test
    val_frac = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_frac), random_state=seed)
    val_local, test_local = next(sss2.split(temp_idx, labels[temp_idx]))
    val_idx = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    splits = DatasetDict({
        "train": dataset.select(train_idx.tolist()),
        "validation": dataset.select(val_idx.tolist()),
        "test": dataset.select(test_idx.tolist()),
    })

    print(f"Splits — train: {len(splits['train'])}, "
          f"val: {len(splits['validation'])}, test: {len(splits['test'])}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(save_dir))

    return splits
