from .module_gradient_descent import gradient_descent


# ==================================================================================================================== #
# Градиентный спуск

def gradient_descent_constant(f, x0, lr=0.01, eps=1e-6, num_iterations=1000, minimum=0, apply_min=False):
    """
    Градиентный спуск с постоянным шагом.

    Аргументы:
    f -- функция
    x0 -- начальная точка
    lr -- постоянный коэффициент обучения (learning rate)
    num_iterations -- количество итераций

    Возвращает:
    gradient_descent(...)
    """

    def const_lr(*ignored):
        return lr

    return gradient_descent(f, x0, const_lr, num_iterations, eps, minimum=minimum, apply_min=apply_min)

# ==================================================================================================================== #
