import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits import mplot3d
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150

def f(x, y):
    return x**2-y**2
def grad_f(x, y, name='G'):
    if name=="S":
        return [-2 * x / (np.abs(x) + 0.000001), 2 * y / (np.abs(y) + 0.000001)]
    else:
        return [2*x, -2*y]
# 定义初始值
x1, x2 = 0.1, 0.1
# 定义步长 
eta = [0.3, 0.3, 0.3, 0.3]
step = np.zeros([4, 20, 2])
# 迭代求解过程
def bound(x, x1, x2):
    return x
    if x>x2+0.5:
        return x+0.5 
    if x<x1-0.5:
        return x-0.5
    else:
        return x

init = [[-3, 0], [3, 0], [-3, 0.1], [-3, 0.2]]
for itr in range(4):
    x1, x2 = init[itr]
    for t in range(20):
        step[itr, t, :] = np.array([bound(x1, -3, 1), bound(x2, -2, 2)])
        g1, g2 = grad_f(x1, x2)
        x1 = x1 - 0.1 * g1 
        x2 = x2 - 0.1 * g2 
        #print("f({:.2f}, {:.2f})={:.2f}".format(x1, x2, f(x1, x2)))
        

#ax = plt.axes(projection='3d') 
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
x1, x2 = np.meshgrid(x1, x2)
x3 = f(x1, x2)

C = plt.contour(x1, x2, x3, 16,colors='black')

for itr in range(4):
    line = step[itr]
    lx, ly = line[:, 0], line[:, 1]
    plt.plot(lx, ly, marker="$%d$"%(itr+1), label="$x_0=(%.2f, %.2f)$"%(init[itr][0], init[itr][1]), alpha=0.5)


plt.axis("equal")
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.clabel(C,inline=1,fontsize=10)
plt.show()