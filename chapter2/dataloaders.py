import numpy as np
import torch
from torch.utils.data import Dataset


class Simple1DRegressionDataSet(Dataset):
    def __init__(self, features: np.array, targets: np.array) -> None:
        """
        Data Loader for py-torch training.
        :param features: {array-like}, shape = [n_examples, n_features] Training vectors,
            where n_examples is the number of examples and n_features is the number of features.
        :param targets: array-like, shape = [n_examples] Target values.
        """
        super(Simple1DRegressionDataSet, self).__init__()
        self.features = features.reshape(-1, 1)
        self.targets = targets.reshape(-1, 1)

    def __getitem__(self, item: int) -> tuple:
        return (
            torch.tensor(self.features[item, :], dtype=torch.float32),
            torch.tensor(self.targets[item, :], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self.features.shape[0]
