## Generate a contour plot
# Import some other libraries that we'll need
# matplotlib and numpy packages must also be installed
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150


# define objective function
def f(x1, x2):
    u = 2.5*x1
    v = 2.5*x2
    return u * u + 5 * v * np.sin(v) + 5 * v

# Start location
x_start = [-2, 2]

# Design variables at mesh points
i1 = np.arange(-3.0, 3.0, 0.05)
i2 = np.arange(-3.0, 3.0, 0.05)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
fm = f(x1m, x2m)


init = [[-2, 2], [2, 2], [0, 0], [-1, -1]]
step = []
for itr in range(4):
# 迭代次数
N = 50
# 转移次数
n_trans = 0.0
# 最大概率
p_max = 0.99
# 终止概率
p_min = 0.001
# 初始温度
T = -1.0/np.log(p_max)
# Final temperature
T_min = -1.0/np.log(p_min)
# 每次循环
beta = (T_min/T)**(1.0/(N-1.0))
# 初始化取值
x1, x2 = init[itr]
n_trans += 1.0
t = T 
# 能量均值
avg_E = 0.0
for i in range(N):
    for j in range(60):
        # 产生新的解
        x1t = x1 + (np.random.random() - 0.5) * 1 * t
        x2t = x2 + (np.random.random() - 0.5) * 1 * t
        # 使得解在求解空间中
        x1t = max(min(x1t,3.0),-3.0)
        x2t = max(min(x2t,3.0),-3.0)
        # 计算能量增量
        delta_E = f(x1t, x2t)-f(x1, x2)
        if delta_E > 0:
            # 此时解使得结果变差，需要根据一定方式决定是否接受新解
            if (i==0 and j==0): avg_E = delta_E
            print(delta_E, avg_E, t)
            # 根据Metropolis准则决定是否接受新样本
            p = np.exp(-delta_E/(avg_E * t))
            if (np.random.random()<p):
                # 接受
                accept = True
            else:
                # 拒绝这个新的差解
                accept = False
        else:
            # 新解使得结果变好，直接接受新解。
            accept = True
        if (accept==True):
            # 对接进行更新
            x1, x2 = x1t, x2t 
            # increment number of accepted solutions
            n_trans += 1.0
            # 更新能量
            avg_E = (avg_E * (n_trans-1.0) +  delta_E) / n_trans
    # 在下次循环中降低温度
    t = beta * t
        # 记录
        
    step.append(result)

step = np.array(step)


for itr in range(4):
    line = step[itr]
    lx, ly = line[:, 0], line[:, 1]
    plt.plot(lx, ly, marker="$%d$"%(itr+1), label="$x_0=(%.2f, %.2f)$"%(init[itr][0], init[itr][1]), alpha=0.5)


plt.axis("equal")
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
x1, x2 = np.meshgrid(x1, x2)
x3 = f(x1, x2)

C = plt.contour(x1, x2, x3, 16,colors='black')
plt.clabel(C,inline=1,fontsize=10)
plt.show()
