import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

x = np.linspace(0, 2*np.pi, 300)
y1 = np.sin(x*2) 
y2 = np.sin(40*x) * 0.1
y = y1+y2
plt.plot(x, y, c="k", lw=1, label="原始信号")
plt.scatter(x, y, s=10, c="r", label="采样点")
plt.xlabel("时间（s）")
plt.legend()
plt.show()