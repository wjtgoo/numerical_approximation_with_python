import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import sympy as sym

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 拉格朗日函数
def lagrange(x, X, Y):
    if len(X) != len(Y):
        raise ValueError("输入的插值节点X变量与Y变量长度不对应！")
    if type(x) == int:
        x = [x]
    Y = np.array(Y)
    n = len(X)
    y = []
    # 求所有待估计点
    for xi in x:
        # 定义l保存l_1,l_2,...l_n
        l = np.ones(n)
        for i in range(n):
            for ii in range(n):
                if i != ii:
                    l[i] = l[i] * (xi - X[ii]) / (X[i] - X[ii])
        yi = Y @ l
        y.append(yi)
    y = np.array(y)
    return y


# Hermite插值
def taylor_expan(func, xx, num_terms):
    '''
    泰勒展开函数
    func:符号变量函数，自变量为x
    num_terms:展开的次数
    xx:展开的位置
    '''
    x = sym.Symbol('x')
    sums = 0
    for i in range(num_terms + 1):
        # 求i次导数
        numerator = func.diff(x, i)
        # 导数在xx点的值（泰勒展开分子）
        numerator = numerator.evalf(subs={x: xx})
        # i的阶乘
        denominator = np.math.factorial(i)
        # 累加项
        sums += numerator / denominator * (x - xx) ** i
    return sums


def A_(x, X, m):
    '''
    x:符号变量
    X:插值节点
    m:给定的mi阶导信息
    '''
    x = sym.Symbol('x')
    out = 1
    for i, mi in enumerate(m):
        out *= (x - X[i]) ** mi
    return out


def hermite(x_i, X, Y):
    '''
    x_i:待插值节点
    X:插值节点
    Y:插值节点值
    '''
    x_i = np.array(x_i)
    X = np.array(X)
    Y = np.array(Y)
    # 插值点数
    k = len(X)
    # 给定的mi阶导信息
    flag = np.isnan(Y) == False
    m = flag.sum(axis=0)
    # 求mi-j-1阶泰勒展开
    x = sym.Symbol('x')
    A = A_(x, X, m)
    hermite_out = 0
    for i in range(k):
        for j in range(m[i]):
            func_1 = A / (x - X[i]) ** m[i]
            func_2 = (x - X[i]) ** m[i] / A
            taylor = taylor_expan(func_2, X[i], m[i] - j - 1)
            fji = Y[j, i]
            hermite_out += func_1 * fji * (x - X[i]) ** j / np.math.factorial(j) * taylor
    out = sym.lambdify('x', hermite_out, "numpy")
    return out(x_i)


# 原函数值
X = np.linspace(-1, 1, 200)
Y = 1 / (1 + 25 * X ** 2)
'''
N = 20
for n in range(N):
    # Lagrange 插值
    X1 = np.linspace(-1, 1, n + 2)
    Y1 = 1 / (1 + 25 * X1 ** 2)
    x1 = np.linspace(-1, 1, 200)
    y1 = lagrange(x1, X1, Y1)
    # Hermite 插值
    X2 = X1
    Y_2 = -50*X2 / (1 + 25 * X2 ** 2)**2
    Y2 = np.array([Y1,Y_2])
    x2 = np.linspace(-1, 1, 200)
    y2 = hermite(x2,X2,Y2)
    plt.plot(X, Y, label='Origin')
    plt.plot(x1, y1, label=str(n+1) + '次Lagrange插值')
    plt.plot(x2, y2, label=str(2*n+1) + '次Hermite插值')
    plt.legend(loc='best')
    plt.show(block=False)
    plt.pause(1)
    plt.clf()
'''
N = 20
for n in range(N):
    # Lagrange 插值
    X1 = np.linspace(-1, 1, n + 2)
    Y1 = 1 / (1 + 25 * X1 ** 2)
    x1 = np.linspace(-1, 1, 200)
    y1 = lagrange(x1, X1, Y1)
    # 切比雪夫点的Lagrange插值
    t = np.linspace(0, np.pi, n + 2)
    X2 = np.cos(t)
    Y2 = 1 / (1 + 25 * X2 ** 2)
    y2 = lagrange(x1, X2, Y2)
    plt.plot(X, Y, label='Origin')
    plt.plot(x1, y1, label=str(n + 1) + '次Lagrange插值')
    plt.plot(x1, y2, label=str(n + 1) + '次切比雪夫节点的L插值')
    plt.legend(loc='best')
    plt.show(block=False)
    plt.pause(1)
    plt.clf()
