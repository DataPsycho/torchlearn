import torch.nn as nn
import torch


class LinearRegression1D:
    def __init__(self, in_feature=1, out_feature=1):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.device = torch.device("cuda")
        self.loss_func = nn.MSELoss()

    def set_device(self, device: str):
        devices = ["cuda", "cpu"]
        if device not in devices:
            raise ValueError("Only {} is valid as device name".format(" and ".join(devices)))
        self.device = torch.device(device)

    def get_device(self):
        return self.device

    def get_loss_func(self):
        return self.loss_func

    def __call__(self, *args, **kwargs):
        model = nn.Linear(self.in_feature, self.out_feature)
        return model
