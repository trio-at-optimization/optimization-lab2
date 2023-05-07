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
# Визуализация линий уровня градиентов

def print_lines_grad(file_info_3d, list_result, list_label, title='Градиентный спуск на уровнях функции', filename='',
                     filename_extension='.png', dpi=1024):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for i in range(len(list_result)):
        levels = np.unique(sorted([file_info_3d.f(p) for p in list_result[i]]))
        ax.contour(file_info_3d.X, file_info_3d.Y, file_info_3d.Z, levels=levels, colors='red', antialiased=True, linewidths=1.0)

    for i in range(len(list_result)):
        x = list_result[i][:, 0]
        y = list_result[i][:, 1]
        ax.plot(x, y, marker='.', markersize=10, markerfacecolor='black', label=list_label[i], linewidth=2)
        print(
            f'{list_label[i]:15} ==> '
            f'{file_info_3d.f(list_result[i][-1]):10f} in [{list_result[i][-1][0]:10f}, {list_result[i][-1][1]:10f}]')

    # Добавление заголовка и подписей осей
    if title != '':
        plt.title(title)

    # Добавление легенды
    if len(list_label) > 0:
        plt.legend(loc='upper left')

    if filename != '':
        plt.savefig(filename + '_lines' + filename_extension, dpi=dpi, bbox_inches=0, transparent=True)

    plt.show()


# ==================================================================================================================== #
# Анимированная визуализация линий уровня градиентов

def print_lines_grad_animated(file_info_3d, list_result, list_label, interval=100, frames=-1):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    def frame(w):
        ax.clear()

        for i in range(len(list_result)):
            levels = np.unique(np.sort(file_info_3d.f(list_result[i][:frames].T)))
            ax.contour(file_info_3d.X, file_info_3d.Y, file_info_3d.Z, levels, colors='red', antialiased=True, linewidths=1.0)

        for i in range(len(list_result)):
            x = list_result[i][:w, 0]
            y = list_result[i][:w, 1]
            ax.plot(x, y, marker='.', markersize=10, markerfacecolor='black', label=list_label[i], linewidth=2)

        ax.legend(loc='upper left')

        return ax

    plt.close()
    if frames == -1 or frames > len(list_result[0]):
        frames = len(list_result[0])

    return FuncAnimation(fig, frame, interval=interval, frames=frames, blit=False, repeat=True)

# ==================================================================================================================== #
