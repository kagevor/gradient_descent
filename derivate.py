from sympy import *


def f_der(f):
    x = Symbol('x')
    func = eval(f)
    func_der = func.diff(x)
    return func, func_der
