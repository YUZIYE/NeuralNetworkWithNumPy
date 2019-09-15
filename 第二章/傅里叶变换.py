import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal


plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
Nsk = 151
plt.subplot(321)
x = np.linspace(0, 2*np.pi, 300)
y1 = np.sin(x*2) 
y2 = np.sin(60*x) * 0.1
yy = y1+y2
plt.plot(x, yy, c="k", lw=1, label="原始信号")
#plt.scatter(x, yy, s=10, c="r", label="采样点")
#plt.xlabel("时间（s）")
plt.xticks([], [])
plt.legend(loc="upper right")
plt.subplot(322)
x = np.linspace(0, 1/(2*np.pi)*300, 300)

y = np.abs(np.fft.fft(yy))
l = len(x)
plt.plot(x[:Nsk], y[:Nsk], c="k", lw=1, label=r"频谱幅值$|y_i|$")
#plt.scatter(x, y, s=10, c="r", label="采样点")
#plt.xlabel("频率（Hz）")
plt.xticks([], [])
plt.legend(loc="upper right")


b, a = signal.butter(8, 0.125, 'lowpass') 
yyflt = signal.filtfilt(b,a,yy)

plt.subplot(324)
x = np.linspace(0, 1/(2*np.pi)*300, 300)
y = np.abs(np.fft.fft(yyflt))
l = len(x)
plt.plot(x[:Nsk], y[:Nsk], c="k", lw=1, label=r"低通滤波后频谱幅值$|y_i|$")
plt.ylim([0, 155])
#plt.scatter(x, y, s=10, c="r", label="采样点")
plt.xticks([], [])
plt.legend(loc="upper right")

plt.subplot(323)
x = np.linspace(0, 2*np.pi, 300)

plt.plot(x, yyflt, c="k", lw=1, label=r"低通滤波后波形")
#plt.scatter(x, y, s=10, c="r", label="采样点")
plt.xticks([], [])
plt.legend(loc="upper right")

b, a = signal.butter(8, 0.4, 'highpass') 
yyflt = signal.filtfilt(b,a,yy)

plt.subplot(326)
x = np.linspace(0, 1/(2*np.pi)*300, 300)
y = np.abs(np.fft.fft(yyflt))
l = len(x)
plt.plot(x[:Nsk], y[:Nsk], c="k", lw=1, label=r"高通滤波后频谱幅值$|y_i|$")
#plt.scatter(x, y, s=10, c="r", label="采样点")
plt.xlabel("频率（Hz）")
plt.ylim([0, 155])
plt.legend(loc="upper right")

plt.subplot(325)
x = np.linspace(0, 2*np.pi, 300)

plt.plot(x, yyflt, c="k", lw=1, label=r"高通滤波后波形")
#plt.scatter(x, y, s=10, c="r", label="采样点")
plt.xlabel("时间（s）")
plt.legend(loc="upper right")

plt.show()