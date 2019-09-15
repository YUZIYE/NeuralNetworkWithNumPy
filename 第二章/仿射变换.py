
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


ax = []
fig1 = plt.figure()


t = np.linspace(0, 2*np.pi, 30)
X = np.zeros([90, 2])
X[:30, 0] = np.sin(t)
X[:30, 1] = np.cos(t) 
X[30:60, 0] = t - np.pi 
X[30:60, 1] = 0.3*X[30:60, 0]**2 - 1
X[60:, 0] = t - np.pi 
X[60:, 1] = t - np.pi 

vec = np.array([[1, 0], [0, 1]])

ax.append(fig1.add_subplot(gs[0, 0]))
ax[-1].scatter(X[:, 0], X[:, 1], c="b", label="坐标x下的点")
ax[-1].plot([0, vec[0, 0]], [0, vec[0, 1]], lw=3, c="b")
ax[-1].plot([0, vec[1, 0]], [0, vec[1, 1]], lw=3, c="b", label="x坐标空间下的坐标向量")
ax[-1].axis("equal")
ax[-1].set_xlabel("$x_1$")
ax[-1].set_ylabel("$x_2$")
ax[-1].set_xlim([-5, 5])
ax[-1].set_ylim([-5, 5])
ax[-1].legend()
ax.append(fig1.add_subplot(gs[0, 1]))
X = X.dot([[1.3, 1], [-0.6, 1]])
vec2 = vec.dot([[1.3, 1], [-0.6, 1]])
ax[-1].scatter(X[:, 0], X[:, 1], c="r", label="坐标y下的点")
ax[-1].plot([0, vec2[0, 0]], [0, vec2[0, 1]], lw=3, c="b")
ax[-1].plot([0, vec2[1, 0]], [0, vec2[1, 1]], lw=3, c="b", label="x坐标空间下的坐标向量")
ax[-1].plot([0, vec[0, 0]], [0, vec[0, 1]], lw=3, c="r")
ax[-1].plot([0, vec[1, 0]], [0, vec[1, 1]], lw=3, c="r", label="y坐标空间下的坐标向量")
ax[-1].axis("equal")
ax[-1].set_xlabel("$y_1$")
ax[-1].set_ylabel("$y_2$")
ax[-1].set_xlim([-5, 5])
ax[-1].set_ylim([-5, 5])
ax[-1].legend()
plt.show()