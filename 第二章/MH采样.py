"""
用于演示Metropolis-Hastings算法
"""
import numpy as np 
def p(x):
    """
    所需概率密度函数
    """
    return np.exp(-np.sum(x**2)/2)/np.pi/2
def cal_alpha(x1, x2):
    return np.min([p(x2)/p(x1), 1])

# 产生均匀分布的样本
x = np.random.random([1000, 2]) * 10 - 5
# 选择后样本
x_accept = []
x_reject = []
c = 100/np.pi/2
x_accept.append(x[0])
for itr in range(len(x)-1):
    alpha = cal_alpha(x_accept[-1], x[itr+1])
    s = np.random.random() 
    if s < alpha:
        x_accept.append(x[itr+1])
    else:
        x_accept.append(x_accept[-1])
        x_reject.append(x[itr+1])

x_accept = np.array(x_accept)
x_reject = np.array(x_reject)

# 绘图
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
plt.title("MH采样结果")
plt.scatter(x_accept[:, 0], x_accept[:, 1], marker='o', color="#000099", alpha=1.0, label="接受样本")
plt.scatter(x_reject[:, 0], x_reject[:, 1], marker='+', color="#990000", alpha=0.5, label="拒绝样本")
plt.plot(x_accept[:, 0], x_accept[:, 1], color="#000099", alpha=0.3, label="转移过程")
plt.legend()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show() 