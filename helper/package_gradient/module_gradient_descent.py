from .module_gradient import gradient

# --------------------------------------------------------------------------------------------------- #
# Для расчетов 
import numpy as np


# =================================================================================================== #
# Градиентный спуск

def gradient_descent(f, x0, lr_search_func, num_iterations=1000, eps=1e-6, step_size=1, minimum=0, apply_min=False):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    ----------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    ----------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    def line_search(x_dot, d):
        def fd(alpha):
            return f(x_dot - alpha * d)
        return lr_search_func(fd, 0, 1, eps)

    x = np.copy(x0)
    points = np.array([x])
    for i in range(num_iterations):
        if apply_min and abs(f(x) - minimum) < eps:
            break

        grad_x = gradient(f, x)
        x = x - grad_x * line_search(x, grad_x) * step_size
        points = np.vstack([points, x])
    return points

# =================================================================================================== #
