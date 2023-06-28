# =================================================================================================== #
# Для графиков
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------- #
# Для расчетов
import numpy as np

# --------------------------------------------------------------------------------------------------- #
# Для анимаций
from matplotlib.animation import FuncAnimation
from matplotlib import rc

# --------------------------------------------------------------------------------------------------- #
# Для больших анимаций

import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 1000.0

rc('animation', html='jshtml')
# Для точки НА графике
plt.style.use('fivethirtyeight')


# =================================================================================================== #
# 3D визуализация функции

def print_f(file_info_3d, elev=30, azim=60):
    # Создание фигуры и трехмерной оси
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Установка угол обзора
    ax.view_init(elev=elev, azim=azim)

    # Построение поверхности
    ax.plot_surface(file_info_3d.X, file_info_3d.Y, file_info_3d.Z)

    # Построение начальной точки
    ax.plot(file_info_3d.x0[0], file_info_3d.x0[1], file_info_3d.f(file_info_3d.x0), 'ro', label='Начальная точка')

    # Установка отступа между графиком и значениями осей
    ax.tick_params(pad=10)

    # Добавление легенды
    plt.legend(loc='upper left')

    # Установка размера шрифта для подписей осей
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # Добавление заголовка и подписей осей
    plt.title('График функции с начальной точкой')
    ax.set_xlabel('Ось X', labelpad=20.0)
    ax.set_ylabel('Ось Y', labelpad=20.0)
    ax.set_zlabel('Ось f(x, y)', labelpad=20.0)

    # Отображение графика
    plt.show()
    return


# =================================================================================================== #
# 3D анимированная визуализация функции

def print_f_animated(file_info_3d, interval=100, elev=30, st_azim=80, delta=5):
    # plt.title('График функции с начальной точкой')
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')

    def frame(w):
        ax.clear()

        # угол обзора
        azim = (w * delta + st_azim) % 360
        ax.view_init(elev=elev, azim=azim)

        # Построение поверхности
        ax.plot_surface(file_info_3d.X, file_info_3d.Y, file_info_3d.Z)

        # Построение начальной точки
        label = 'elev=' + str(elev) + ', azim=' + str(azim)
        ax.plot(file_info_3d.x0[0], file_info_3d.x0[1], file_info_3d.f(file_info_3d.x0), 'ro', markersize=3, label=label)

        # Установка отступа между графиком и значениями осей
        ax.tick_params(pad=10)

        # Установка размера шрифта для подписей осей
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)

        # Добавление заголовка и подписей осей
        ax.set_xlabel('Ось X', labelpad=20.0)
        ax.set_ylabel('Ось Y', labelpad=20.0)
        ax.set_zlabel('Ось f(x, y)', labelpad=20.0)

        ax.legend(loc='upper left')

        return ax

    plt.close()

    frames = np.ceil(360 / delta).astype(int)

    return FuncAnimation(fig, frame, interval=interval, frames=frames, blit=False, repeat=True)

# =================================================================================================== #
