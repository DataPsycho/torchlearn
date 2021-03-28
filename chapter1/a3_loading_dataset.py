from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml
import torch
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


class SimpleDataSet(Dataset):
    def __init__(self, inputs, targets):
        super(SimpleDataSet, self).__init__()
        self.X = inputs
        self.y = targets

    def __getitem__(self, index):
        input_x = torch.tensor(self.X[index, :], dtype=torch.float32)
        target_y = torch.tensor(int(self.y[index]), dtype=torch.int64)
        return input_x, target_y

    def __len__(self):
        return self.X.shape[0]


dataset = SimpleDataSet(X.to_numpy(), y)

print(len(dataset))
example, label = dataset[0]

plt.imshow(example.reshape(28, 28))

# Train Test split
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, test_size))

print(" {} examples for training and {} for testing".format(len(train_dataset), len(test_dataset)))