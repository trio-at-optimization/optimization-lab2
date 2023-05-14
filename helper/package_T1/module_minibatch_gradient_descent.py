# Для расчетов
import numpy as np


# ==================================================================================================================== #
# Градиентный спуск

def custom_gradient_descent_with_lr_scheduling(
                                               f
                                               , gradient, x0
                                               , lr_scheduling_func
                                               , initial_lr=1
                                               , num_iterations=1000
                                               , eps=1e-6
                                               , minimum=0.0
                                               , apply_min=False
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
    for i in range(num_iterations):
        if apply_min and abs(f(x) - minimum) < eps:
            break

        grad_x = gradient(f, x)
        x = x - grad_x * lr_scheduling_func(i, initial_lr)
        points.append(x.copy())
    return np.array(points)

# ==================================================================================================================== #
