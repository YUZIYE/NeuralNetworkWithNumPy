
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
gs = gridspec.GridSpec(1, 2)
gs.update(hspace=0.3, wspace=0.3)


def f(x1, x2):
    """
    定义函数
    输入：x1，x2函数自变量
    输出：函数值
    """
    return x1**2 + 2*x1 + x2**2 - 1 


x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200) 
xx1 = x1 
yy1 = -xx1+1
x1, x2 = np.meshgrid(x1, x2) 
x3 = f(x1, x2)


ax = []
fig1 = plt.figure()
ax.append(fig1.add_subplot(gs[0, 0]))
C = ax[-1].contour(x1, x2, x3, 8,colors='black', label="函数值")
yy2 = -np.ones_like(xx1) * 4
#ax[-1].scatter([], [], c="k", marker="o", alpha=0, label="问题1")
ax[-1].plot(xx1, yy1, c='k', label="$x_1+x_2-1=0$")
ax[-1].axis("equal")
ax[-1].fill_between(xx1, yy1, yy2, facecolor="lightgray", label="问题1可行域")
ax[-1].set_xlim([-3, 3])
ax[-1].set_ylim([-3, 3])
ax[-1].legend()
plt.clabel(C,inline=1,fontsize=10)
ax.append(fig1.add_subplot(gs[0, 1]))
#ax[-1].scatter([], [], c="k", marker="o", alpha=0, label="问题2")
C = ax[-1].contour(x1, x2, x3, 8,colors='black', label="函数值")
yy2 = np.ones_like(xx1) * 4
ax[-1].plot(xx1, yy1, c='k', label="$x_1+x_2-1=0$")
ax[-1].axis("equal")
ax[-1].fill_between(xx1, yy1, yy2, facecolor="lightgray", label="问题2可行域")
ax[-1].set_xlim([-3, 3])
ax[-1].set_ylim([-3, 3])
ax[-1].legend()
plt.clabel(C,inline=1,fontsize=10)

plt.show()