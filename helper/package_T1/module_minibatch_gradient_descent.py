# Для расчетов
import numpy as np

from ..package_T2 import constant_lr_scheduling

# ==================================================================================================================== #
# Градиентный спуск

def custom_gradient_descent_with_lr_scheduling(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x)
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr)

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points


# ==================================================================================================================== #

class minibatch_library:
    def mse_loss(self, x):
        y_pred = np.dot(self.X, x[0])
        mse = np.mean((self.y - y_pred) ** 2)
        return mse

    def mse_loss_grad(self, x):
        # Choose n random data points from the training set without replacement
        indices = np.random.choice(self.X.shape[0], self.batch_size, replace=False)
        X_batch = self.X[indices, :]
        y_batch = self.y[indices]

        # Compute the gradient of the MSE loss with respect to x for the chosen data points
        y_pred = np.dot(X_batch, x)
        grad = 2 * np.dot(X_batch.T, (y_pred - y_batch))

        # Sum values in rows of grad and divide by n
        grad_mean = np.sum(grad, axis=1) / self.batch_size

        return grad_mean

    def __init__(self, X, y, batch_size=2, method='mse'):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]

        if method == 'mse':
            self.f = self.mse_loss
            self.grad = self.mse_loss_grad
        else:
            print('method not found')

    def gradient_descent(self, x0, lr_scheduling_func=constant_lr_scheduling, initial_lr=0.001, max_epochs=1000,
                         eps=1e-5, minimum=0.0, apply_min=False, apply_value=True):
        """
        Стохастический градиентный спуск для поиска минимума функции.

        Аргументы:
            x0 (list): Начальную точка, с которой начинается поиск.
            initial_lr (float): learning_rate - Начальная скорость обучения или шаг градиентного спуска.
            max_epochs (int): Максимальное количество эпох или итераций для выполнения алгоритма.
            minimum (float): Минимум функции.
            epsilon (float): Малое число, используемое как критерий останова для алгоритма.
        Возвращает:
            Список всех точек, посещенных во время алгоритма.
        """
        return custom_gradient_descent_with_lr_scheduling(self.f, self.grad, x0, lr_scheduling_func, initial_lr,
                                                          max_epochs, eps, minimum, apply_min, apply_value)

    def get_loss_history(self, results):
        loss_history = []

        for i in range(len(results)):
            loss_history.append(self.f(results[i]))

        return loss_history

# ==================================================================================================================== #
