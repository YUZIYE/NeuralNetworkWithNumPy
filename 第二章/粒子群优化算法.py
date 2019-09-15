# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def f(x1, x2):
    u = x1 * 2.5
    v = x2 * 2.5
    return u * u + 5 * v * np.sin(v) + 5 * v 

# 定义粒子个数
N = 20 
# 定义惯性因子
w = 0.6
# 定义C1，C2 
c1, c2 = 2, 2 
# 初始化位置
x = np.random.uniform(-3, 3, [N, 2]) 
# 初始化速度
v = np.random.uniform(-1, 1, [N, 2])
# 个体最佳位置
p_best = np.copy(x) 

fitness = f(x[:, 0], x[:, 1])
fitness = np.expand_dims(fitness, 1)
# 群体最佳位置
g_best = p_best[np.argmin(fitness)]
store = np.zeros([N, 20, 2])
for step in range(20):
    # 计算速度v
    store[:, step, :] = x
    r1, r2 = np.random.random([N, 1]), np.random.random([N, 1])
    v = w * v + (1-w)*(c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)) 
    # 更新位置
    x = x + v 
    x = np.clip(x, -3, 3)
    # 计算适应度
    fitness_new = f(x[:, 0], x[:, 1]) 
    fitness_new = np.expand_dims(fitness_new, 1)
    fit = np.concatenate([fitness, fitness_new], 1) 
    fitness = fitness_new
    # 计算个体最优解
    p_best_for_sel = np.concatenate([
        np.expand_dims(x, 1), 
        np.expand_dims(p_best, 1)], 1) 
    p_best = p_best_for_sel[[i for i in range(N)], np.argmin(fit, 1), :]
    fit_p = f(p_best[:, 0], p_best[:, 1])
    # 计算全局最优解
    g_best = x[np.argmin(fitness[:, 0])]
    print(g_best)


for itr in range(N):
    plt.plot(store[itr, :, 0], store[itr, :, 1], marker="$%d$"%(itr+1), label="粒子%d"%(itr+1)) 

x1 = np.linspace(-3, 3, 200)  
x2 = np.linspace(-3, 3, 200)   
x1, x2 = np.meshgrid(x1, x2)   
x3 = f(x1, x2)

C = plt.contour(x1, x2, x3, 16,colors='black') 
plt.clabel(C,inline=1,fontsize=10)
plt.axis("equal")
plt.legend()
plt.show()
plt.show()