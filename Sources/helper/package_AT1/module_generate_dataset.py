import torch
import numpy as np
import matplotlib.pyplot as plt


# ==================================================================================================================== #


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

        within_radius = (x_rand - X) ** 2 + (y_rand - Y) ** 2 <= radius ** 2
        if torch.any(within_radius):
            dataset_X.append([x_rand.item()])
            dataset_Y.append([y_rand.item()])

    return np.array(dataset_X), np.array(dataset_Y)


# ==================================================================================================================== #


def print_generated_dataset(X, Y, dataset_X, dataset_Y, radiusX=None, radiusY=None):
    plt.style.use('default')
    _ = plt.figure(figsize=(8, 8))

    x_min = min(X)
    x_max = max(X)
    y_min = min(Y)
    y_max = max(Y)
    if radiusX:
        x_min -= radiusX
        x_max += radiusX
    if radiusY:
        y_min -= radiusY
        y_max += radiusY

    # Задаем координаты вершин прямоугольника
    x = [x_min, x_min, x_max, x_max, x_min]
    y = [y_min, y_max, y_max, y_min, y_min]

    # Рисуем прямоугольник
    plt.plot(x, y)

    plt.scatter(dataset_X, dataset_Y, color='gray', alpha=0.5, s=20.8, antialiased=True)
    plt.plot()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.plot(X, Y, label='Real', color='lime', antialiased=True, linewidth=1.7)

    plt.legend()
    plt.show()

# ==================================================================================================================== #
