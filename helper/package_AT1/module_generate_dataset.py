import torch
import numpy as np


def generate_dataset(X, Y, dots_count, radius):
    dataset_X = []
    dataset_Y = []

    x_min = min(X) - radius
    y_min = min(Y) - radius
    x_max = max(X) + radius
    y_max = max(Y) + radius

    method = 'cpu'
    if torch.cuda:
        if torch.cuda.is_available():
            method = 'cuda'

    print(method)
    device = torch.device(method)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    while len(dataset_X) < dots_count:
        x_rand = torch.empty(1).uniform_(x_min, x_max).to(device)
        y_rand = torch.empty(1).uniform_(y_min, y_max).to(device)

        within_radius = (x_rand - X)**2 + (y_rand - Y)**2 <= radius**2
        if torch.any(within_radius):
            dataset_X.append([x_rand.item()])
            dataset_Y.append([y_rand.item()])

    return np.array(dataset_X), np.array(dataset_Y)
