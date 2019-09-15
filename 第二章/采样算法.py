import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy import read 
from scipy import signal 
# define a list of markevery cases to plot
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

import numpy as np 
def p(x):
    """
    所需概率密度函数
    """
    return np.exp(-x**2/2)/np.pi/2
def sampler():
    """
    生成一个随机样本
    """
    # 产生多个样本
    xs = np.random.random([20]) * 10 - 5
    # 计算概率
    ps = p(xs)
    # 计算累计概率
    t = np.cumsum(ps)/np.sum(ps)
    # 产生均匀分布随机数
    u = np.random.random()
    # 根据概率选择样本
    sample_id = int(np.searchsorted(t, np.random.random()))
    return xs[sample_id]
# 给定初始值
x_samples = [] 
n_samples = 5000

for itr in range(n_samples):
    x_samples.append(sampler()) 
x_samples = np.array(x_samples)

plt.hist(x_samples, bins=60, normed=True, color="#000099")
plt.xlabel("x")
plt.ylabel("p(x)")
plt.show()