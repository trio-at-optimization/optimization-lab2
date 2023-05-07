# ==================================================================================================================== #
# Для расчетов 
import numpy as np


# ==================================================================================================================== #
# Класс для хранения информации о 3d функции

class file_info_3d:
    def __init__(self, X=None, Y=None, f=None, x0=None):
        self.X = X
        self.Y = Y
        self.Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)
        self.f = f
        self.x0 = x0

# ==================================================================================================================== #
# Класс для хранения информации о 2d наборе данных


def random_generator(real_weights, real_bias, dots_count=100, variance=1):
    X = np.random.rand(dots_count, len(real_weights))
    y = real_weights * X + real_bias + (np.random.rand(dots_count, len(real_weights)) * 2 * variance - variance)
    return X, y


class file_info_2d_data:
    def __init__(self, real_weights, real_bias, full_loss_f=[], dots_count=100, variance=1, labels_loss=None,
                 current_random_generator=random_generator):
        self.real_weights = real_weights
        self.real_bias = real_bias
        self.X, self.y = current_random_generator(real_weights, real_bias, dots_count, variance)
        self.loss_f = []
        for i in range(len(full_loss_f)):
            self.loss_f.append(lambda value: full_loss_f[i](self.X, self.y, value))
        self.labels_loss = labels_loss

# ==================================================================================================================== #
