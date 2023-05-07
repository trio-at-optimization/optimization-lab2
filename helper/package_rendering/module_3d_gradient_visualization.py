# ==================================================================================================================== #
# Для графиков
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# Для расчетов
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# Для анимаций
from matplotlib.animation import FuncAnimation
from matplotlib import rc

# -------------------------------------------------------------------------------------------------------------------- #
# Для больших анимаций

import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 1000.0

rc('animation', html='jshtml')
# Для точки НА графике
plt.style.use('fivethirtyeight')


# ==================================================================================================================== #
# 3D визуализация градиентов

def print_full_grad(file_info_3d, list_result, list_label, title='Градиентный спуск на графике функции', elev=30, azim=80,
                    filename='', filename_extension='.png', dpi=1024):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    for i in range(len(list_result)):
        x = list_result[i][:, 0]
        y = list_result[i][:, 1]
        z = np.vectorize(lambda x_dot, y_dot: file_info_3d.f(np.array([x_dot, y_dot])))(x, y)
        ax.plot(x, y, marker='.', markersize=10, markerfacecolor='black', zs=z, label=list_label[i], linewidth=2)
        print(
            f'{list_label[i]:15} ==> '
            f'{file_info_3d.f(list_result[i][-1]):10f} in [{list_result[i][-1][0]:10f}, {list_result[i][-1][1]:10f}]')

    ax.plot_surface(file_info_3d.X, file_info_3d.Y, file_info_3d.Z, cmap='Spectral')
    ax.view_init(elev=elev, azim=azim)

    # Установка отступа между графиком и значениями осей
    ax.tick_params(pad=10)

    # Добавление легенды
    if len(list_label) > 0:
        ax.legend(loc='upper left')

    # Установка размера шрифта для подписей осей
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # Добавление заголовка и подписей осей
    if title != '':
        plt.title(title)

    ax.set_xlabel('Ось X', labelpad=20.0)
    ax.set_ylabel('Ось Y', labelpad=20.0)
    ax.set_zlabel('Ось f(x, y)', labelpad=20.0)

    if filename != '':
        plt.savefig(filename + filename_extension, dpi=dpi, bbox_inches=0, transparent=True)

    plt.show()


# ==================================================================================================================== #
# 3D анимированная визуализация градиентов

def print_full_grad_animated(file_info_3d, list_result, list_label, interval=100, frames=-1, elev=30, azim=80):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')

    def frame(w):
        ax.clear()
        for i in range(len(list_result)):
            x = list_result[i][:w + 1, 0]
            y = list_result[i][:w + 1, 1]
            z = np.vectorize(lambda x_dot, y_dot: file_info_3d.f(np.array([x_dot, y_dot])))(x, y)
            ax.plot(x, y, marker='.', markersize=10, markerfacecolor='black', zs=z, label=list_label[i], linewidth=2,
                    markevery=(w, w + 1), cmap='Spectral')

        ax.plot_surface(file_info_3d.X, file_info_3d.Y, file_info_3d.Z)
        ax.view_init(elev=elev, azim=azim)
        if len(list_label) > 0:
            ax.legend(loc='upper left')

        return ax

    plt.close()
    if frames == -1 or frames > len(list_result[0]):
        frames = len(list_result[0])

    return FuncAnimation(fig, frame, interval=interval, frames=frames, blit=False, repeat=True)

# ==================================================================================================================== #
