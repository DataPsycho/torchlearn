import torch
from torch.utils.data import DataLoader
import seaborn as sns

from chapter2.dataloaders import Simple1DRegressionDataSet
from chapter2.datagenerators import generate_1d_data
from chapter2.models import LinearRegression1D
from chapter2.train import train_simple_network


features, targets = generate_1d_data()
sns.scatterplot(x=features, y=targets)

training_loader = DataLoader(Simple1DRegressionDataSet(features=features, targets=targets), shuffle=True)
model_instance = LinearRegression1D()

# is cuda is used it throws error: Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
model_instance.set_device("cpu")

model = model_instance()
loss_func = model_instance.get_loss_func()
device = model_instance.get_device()

train_simple_network(model=model, loss_func=loss_func, training_loader=training_loader, device=device)


def get_prediction():
    with torch.no_grad():
        predictions = model(torch.tensor(features.reshape(-1, 1), device=device, dtype=torch.float32)).cpu().numpy()
    return predictions


pred = get_prediction()
sns.scatterplot(x=features, y=targets, color='blue', label='Data')
sns.lineplot(x=features, y=pred.ravel(), color='red', label='Linear Model')