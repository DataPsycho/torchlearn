import numpy as np


def generate_1d_data():
    features = np.linspace(0, 20, num=200)
    targets = features + np.sin(features) * 2 + np.random.normal(size=features.shape)
    return features, targets
