import numpy as np 
def p(x):
    """
    所需概率密度函数
    """
    return np.exp(-np.sum(x**2, axis=1)/2)/np.pi/2
def dim_sampler(x, dim):
    """
    从某个维度采样
    直接计算多个样本的概率，根据概率选择合适的样本。
    由于仅是一维分布，我们也可以用其他方式进行采样。
    """
    # 产生多个样本
    sp = np.random.random([20]) * 10 - 5
    xs = np.ones([20, len(x)])*np.array(x)
    xs[:, dim] = sp
    # 计算概率
    ps = p(xs)
    # 根据概率选择样本
    t = np.cumsum(ps)/np.sum(ps)
    s = np.random.random()
    sample_id = np.searchsorted(t, s)
    return list(xs[sample_id])
# 给定初始值
x_samples = [[0, 0]]
n_samples = 600
state = x_samples[-1]
for itr in range(n_samples):
    for dim in range(len(state)):
        state = dim_sampler(state, dim)
        x_samples.append(state)
x_samples = np.array(x_samples)
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

plt.subplot(121)
plt.title("坐标变换结果")
x_norm = np.random.normal(0, 1, [1000, 2])
plt.scatter(x_norm[:, 0], x_norm[:, 1])
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.subplot(122)
plt.title("吉布斯采样结果")
plt.scatter(x_samples[:, 0], x_samples[:, 1], marker='o', color="#000099", alpha=1.0)
plt.plot(x_samples[:, 0], x_samples[:, 1], color="#000099", alpha=0.3, label="转移过程")
plt.legend()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show() 