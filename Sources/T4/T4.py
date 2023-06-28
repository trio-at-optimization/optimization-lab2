# Для расчетов
import math
import cProfile
import pstats

import numpy as np
from memory_profiler import profile

class minibatch:
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
        

    def constant_lr_scheduling(epoch, initial_lr):
        return initial_lr

    def gradient_descent(self, gradient_descent_func, 
                         x0, lr_scheduling_func=constant_lr_scheduling, 
                         max_epochs=1000, eps=1e-5, 
                         minimum = 0.0, apply_min=False, apply_value=True):
        """
        Cтохастический градиентный спуск для поиска минимума функции.

        Аргументы:
            x0 (list): Начальную точка, с которой начинается поиск.
            initial_lr (float): learning_rate - Начальная скорость обучения или шаг градиентного спуска.
            max_epochs (int): Максимальное количество эпох или итераций для выполнения алгоритма.
            minimum (float): Минимум функции.
            epsilon (float): Малое число, используемое как критерий останова для алгоритма.
        Возвращает:
            Список всех точек, посещенных во время алгоритма.
        """
        return gradient_descent_func(self.f, self.grad, x0, lr_scheduling_func, num_iterations=max_epochs, eps=eps, minimum=minimum, apply_min=apply_min, apply_value=apply_value)
    
    def get_loss_history(self, results):
        loss_history = []

        for i in range(len(results)):
            loss_history.append(self.f(results[i]))

        return loss_history
    


# ==================================================================================================================== #

# @profile
def custom_gradient_descent_with_lr_scheduling_and_moment(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1e-5
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

    count_ops = 0
    x = np.copy(x0)
    moment = 0.0
    points = [x.copy()]
    value = 0.0
    if apply_value:
        value = f(x)
        count_ops += len(x) * 4 + 1
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break

        grad_x = gradient(x)
        current_lr = lr_scheduling_func(i, initial_lr)
        moment = moment*0.9 - current_lr * grad_x
        new_x = x + moment

        count_ops += 4 + len(grad_x) + 6 + 6 * 500

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points, count_ops

# ==================================================================================================================== #

# @profile
def custom_gradient_descent_with_lr_scheduling_and_nesterov_moment(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1e-5
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

    count_ops = 0
    x = np.copy(x0)
    moment = 0.0
    points = [x.copy()]
    value = 0.0
    if apply_value:
        value = f(x)
        count_ops += len(x) * 4 + 1
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break

        grad_x = gradient(x + moment*0.9)
        current_lr = lr_scheduling_func(i, initial_lr)
        moment = moment*0.9 - current_lr * grad_x
        new_x = x + moment

        count_ops += 4 + len(grad_x) + 6 + 6 * 500

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points, count_ops

# ==================================================================================================================== #

# @profile
def custom_gradient_descent_with_lr_scheduling_and_adagrad(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1e-1
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

    count_ops = 0
    eps_adagrad = 1e-6
    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_adagrad
    if apply_value:
        value = f(x)
        count_ops += len(x) * 4 + 1
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break

        grad_x = gradient(x)
        G = G + grad_x.dot(grad_x)
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_adagrad))

        count_ops += 4 + len(grad_x) + 6 + 6 * 500

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points, count_ops

# ==================================================================================================================== #

# @profile
def custom_gradient_descent_with_lr_scheduling_and_RMSProp (
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1e-1
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

    count_ops = 0
    eps_RMSProp = 1e-8
    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_RMSProp
    if apply_value:
        value = f(x)
        count_ops += len(x) * 4 + 1
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break

        grad_x = gradient(x)
        G = G*0.9 + (1 - 0.9) * (grad_x ** 2)
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_RMSProp))

        count_ops += 4 + len(grad_x) + 6 + 6 * 500

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points, count_ops

# ==================================================================================================================== #

# @profile
def custom_gradient_descent_with_lr_scheduling_and_Adam (
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=5e-1
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
    B1 = 0.9
    B2 = 0.999

    count_ops = 0
    eps_Adam = 1e-8
    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_Adam
    moment = 0.0
    if apply_value:
        value = f(x)
        count_ops += len(x) * 4 + 1
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                count_ops += len(x) * 4 + 2
                break

        grad_x = gradient(x)
        moment = moment*B1 + (1 - B1)*grad_x
        G = G*B2 + (1 - B2) * (grad_x ** 2)

        moment_more = moment / (1 - B1 ** i)
        G_more = G / (1 - B2 ** i)
        new_x = x - lr_scheduling_func(i, initial_lr) * (moment_more) / (math.sqrt(G_more + eps_Adam))

        count_ops += 4 + len(grad_x) + 6 + 6 * 500


        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points, count_ops

# ==================================================================================================================== #

if __name__ == '__main__':
    # gradient_descent_function = custom_gradient_descent_with_lr_scheduling_and_adagrad
    gradient_descent_functions = [custom_gradient_descent_with_lr_scheduling_and_moment,
                                  custom_gradient_descent_with_lr_scheduling_and_nesterov_moment,
                                  custom_gradient_descent_with_lr_scheduling_and_adagrad,
                                  custom_gradient_descent_with_lr_scheduling_and_RMSProp,
                                  custom_gradient_descent_with_lr_scheduling_and_Adam]

    real_weight, real_bias = 2, 0

    dots_count = 500
    variance = 0.5
    X = np.random.rand(dots_count, 1)
    y = real_weight * X + real_bias + (np.random.rand(dots_count, 1) * variance - variance / 2)
    sgd = minibatch(X, y, batch_size=500)
    loss_real = sgd.get_loss_history([[real_weight]])[-1]
    x0 = np.array([0], float)


    for gradient_function in gradient_descent_functions:
        _, count = sgd.gradient_descent(gradient_function, 
                                        x0.copy(), max_epochs=10000, 
                                        eps=loss_real+(loss_real*1e-1), 
                                        apply_min=True, apply_value=True)
    
        print(f'{gradient_function}: {count}\n')
    

