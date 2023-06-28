import math


def constant_lr_scheduling(epoch, initial_lr):
    return initial_lr


def exp_decay_const(epoch, initial_lr):
    return math.exp(-0.05 * (epoch + 10))


def exp_decay(epoch, initial_lr):
    return math.exp(-0.05 * (epoch + 10)) * initial_lr

