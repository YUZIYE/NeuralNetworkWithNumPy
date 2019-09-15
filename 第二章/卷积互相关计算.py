import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal
from matplotlib.gridspec import GridSpec 

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
Nsk = 151
gs = GridSpec(3, 6, wspace=0.3, hspace=0.1)
fig = plt.figure(constrained_layout=True) 

ax = fig.add_subplot(gs[0, 2:])
x = np.linspace(0, 2*np.pi, 300)
y1 = np.sin(x*2) 
y2 = np.sin(60*x) * 0.1
yy = y1+y2
ax.plot(x, yy, c="k", lw=1, label="原始信号")
ax.scatter(x, yy, s=10, c="r", label="采样点")
ax.legend(loc="upper right")
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.set_ylim([-1.5, 1.5])

ax = fig.add_subplot(gs[0, :2])
x = np.linspace(0, 8*np.pi/300,8)
w = np.ones([4])/4
#w[2:] = -1/2
ax.plot(w, c="k", lw=1, label="时间域滤波器")
ax.scatter(np.linspace(0, len(w)-1, len(w)),w, s=10, c="r", label="采样点")

ax.set_xticks([], [])
ax.set_yticks([], [])
ax.legend(loc="upper right")
ax.set_ylim([-1.5, 1.5])

ax = fig.add_subplot(gs[1, 3:])
yabs = np.abs(np.fft.fft(yy))
ax.plot(yabs[:151], c="k", lw=1, label="信号频谱")
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.legend(loc="upper right")

ax = fig.add_subplot(gs[1, :3])
ww = np.zeros(300)
ww[:len(w)] = w
yabs = np.abs(np.fft.fft(ww))
ax.set_xticks([], [])
ax.set_yticks([], [])
ax.plot(yabs[:151], c="k", lw=1, label="滤波器频谱")
ax.legend(loc="upper right")

ax = fig.add_subplot(gs[2,:])
yyflt = np.convolve(yy, w,  mode='same')
ax.set_xticks([], [])
ax.set_yticks([], [])
x = np.linspace(0, 2*np.pi, 300)
ax.plot(yyflt, c="k", lw=1, label="滤波后波形")
#ax.scatter(x, yyflt, s=10, c="r", label="采样点")
ax.legend(loc="upper right")

plt.show()