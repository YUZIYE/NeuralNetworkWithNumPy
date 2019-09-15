import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits import mplot3d
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150

def f(x1, x2):
    """
    定义函数
    输入：x1，x2函数自变量
    输出：函数值
    """
    return x1**2 + 2*x1 + x2**2 - 1 

def grad_f(x1, x2):
    """
    定义函数梯度
    输入：x1，x2函数自变量
    输出：函数梯度值
    """
    return 2 * x1 + 2, 2 * x2 
# 定义初始值
x1, x2 = 0.1, 0.1
# 定义步长 
eta = [2, 1, 0.3, 0.03]
step = np.zeros([4, 10, 2])
# 迭代求解过程
def bound(x, x1, x2):
    return x
    if x>x2+0.5:
        return x+0.5 
    if x<x1-0.5:
        return x-0.5
    else:
        return x


for itr in range(4):
    x1, x2 = 0.1, 0.1 
    for t in range(10):
        step[itr, t, :] = np.array([bound(x1, -3, 1), bound(x2, -2, 2)])
        g1, g2 = grad_f(x1, x2)
        x1 = x1 - eta[itr] * g1 
        x2 = x2 - eta[itr] * g2 
        #print("f({:.2f}, {:.2f})={:.2f}".format(x1, x2, f(x1, x2)))
        

#ax = plt.axes(projection='3d') 
x1 = np.linspace(-3, 1, 100)
x2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(x1, x2)
x3 = f(x1, x2)

C = plt.contour(x1, x2, x3, 8,colors='black')

for itr in range(4):
    line = step[itr]
    lx, ly = line[:, 0], line[:, 1]
    plt.plot(lx, ly, marker="$%d$"%(itr+1), label="$\eta=%.2f$"%eta[itr], alpha=0.5)


plt.axis("equal")
plt.xlim([-3, 1])
plt.ylim([-2, 2])
plt.legend()
plt.clabel(C,inline=1,fontsize=10)
plt.show()