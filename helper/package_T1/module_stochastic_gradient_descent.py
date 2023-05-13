import random
from ..package_gradient.module_gradient import *


# ==================================================================================================================== #

def stochastic_gradient_descent(f, initial_point, learning_rate=0.1, max_epochs=1000, minimum=0.0, epsilon=1e-5,
                                batch_size=1, apply_min=False):
    """
    Стохастический градиентный спуск для поиска минимума функции.

    Аргументы:
        f (function): Изначальная функция.
        grad_fn (function): Функция, которая принимает точку и возвращает градиент в этой точке.
        initial_point (list): Начальную точка, с которой начинается поиск.
        learning_rate (float): Скорость обучения или шаг градиентного спуска.
        max_epochs (int): Максимальное количество эпох или итераций для выполнения алгоритма.
        minimum (float): Минимум функции.
        epsilon (float): Малое число, используемое как критерий останова для алгоритма.
        batch_size (int): кол-во координат по которым вычисляется градиент.
    Возвращает:
        Кортеж, содержащий найденную минимальную точку, значение функции в этой точке и список всех точек,
         посещенных во время алгоритма.
    """

    batch_size = min(batch_size, len(initial_point))

    current_point = initial_point.copy()  # текущая точка, инициализируется начальной точкой
    current_value = f(current_point)  # значение функции в текущей точке
    visited_points = [current_point.copy()]  # список посещенных точек, начинается с начальной точки
    for _ in range(max_epochs):  # цикл по эпохам
        if apply_min and abs(current_value - minimum) < epsilon:
            # если достигнуто достаточно малое значение функции, то останавливаемся
            break
        prev_point = np.copy(current_point)
        for _ in range(batch_size):
            random_index = random.randint(0, len(current_point) - 1)
            # выбираем случайный индекс измерения
            gradient_random_index = fast_gradient(f, current_point,
                                                  random_index)
            # вычисляем градиент в текущей точке в случайном индексе
            current_point[random_index] -= learning_rate * gradient_random_index  # обновляем текущую точку

        new_value = f(current_point)  # вычисляем значение функции в обновленной точке
        if new_value < current_value:
            # если значение функции в обновленной точке меньше, чем в предыдущей, то продолжаем движение
            current_value = new_value
        else:  # если значение функции больше или не изменилось, то возвращаемся к предыдущей точке
            current_point = prev_point
        visited_points.append(current_point.copy())  # добавляем текущую точку в список посещенных
    return current_point, current_value, visited_points  # возвращаем результат работы функции

# ==================================================================================================================== #
