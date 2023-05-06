# --------------------------------------------------------------------------------------------------- #
# Для расчетов 
import numpy as np


# =================================================================================================== #
# Градиент

def gradient(f, x):
    delta = 1e-9
    """
    Функция вычисления градиента в заданной точке с константной точностью

    Аргументы:
    f -- функция
    x -- точка

    Возвращает:
    ans -- градиент функции в точке x
    """

    n = len(x)
    xd = np.copy(x)
    ans = np.zeros(n)

    for i in range(n):
        xd[i] += delta
        ans[i] = np.divide(f(xd) - f(x), delta)
        xd[i] -= delta

    return ans

# =================================================================================================== #