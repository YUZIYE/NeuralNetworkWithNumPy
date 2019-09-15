import numpy as np 

# 生成数据
x = np.random.random([1000, 1])*12-6
d = np.sin(x) + np.random.normal(0, 0.02, [1000, 1])

def mm(inputs, kernel, bias):
    """
    全连接网络 
    inputs:输入向量  
    kernel:可训练参数
    bias:偏置
    """
    return np.dot(inputs, kernel) + bias

def model(x, w1, w2, b1, b2):
    """
    含有一个隐藏层的多层感知器模型
    w1,w2:可训练参数
    b1,b2:偏置
    """
    # 隐藏层
    h = mm(x, w1, b2)
    # ReLU激活函数
    hrelu = np.clip(h, 0, np.inf)
    # 输出层
    y = mm(hrelu, w2, b2)
    return y, h, hrelu

def backward(x, d, w1, w2, b1, b2):
    """
    反向传播求解可训练参数的梯度
    x,d:样本和标签
    w1,w2:可训练参数
    b1,b2:偏置    
    """
    # 计算模型和中间结果
    y, h, hrelu = model(x, w1, w2, b1, b2)
    # 定义MSE为损失函数
    loss = np.mean((y-d)**2) 
    # 计算反向传播误差
    err1 = 2 * (y-d) / len(x)
    # 计算输出层可训练参数导数
    dw2 = np.dot(hrelu.T, err1) 
    db2 = np.sum(err1, axis=0) 
    # 计算输出层反向传播误差
    err2 = err1.dot(w2.T)
    mask = (h > 0).astype(np.float64)
    # 计算激活函数层反向传播误差
    err3 = err2 * mask
    # 计算隐藏层可训练参数
    dw1 = np.dot(x.T, err3) 
    db1 = np.sum(err2, axis=0) 
    return loss, dw1, dw2, db1, db2 

# 定义初始值
w1 = np.random.random([1, 1024])*0.02 - 0.01
b1 = np.zeros(1024)
w2 = np.random.random([1024, 1])*0.02 - 0.01
b2 = np.zeros(1)
# 定义学习率
eta = 0.01
# 迭代
for itr in range(20000):
    idx = np.random.randint(0, len(x), 128)
    inx = x[idx]
    ind = d[idx]
    ls, dw1, dw2, db1, db2 = backward(inx, ind, w1, w2, b1, b2)
    w1 -= eta * dw1 
    w2 -= eta * dw2 
    b1 -= eta * db1 
    b2 -= eta * db2 
    print(itr, ls)

import matplotlib.pyplot as plt 
# 测试绘图 
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

testx = np.linspace(-6, 6, 1000).reshape([1000, 1])
predict, _, _ = model(testx, w1, w2, b1, b2) 
plt.scatter(x, d, c="b", label="数据点", alpha=0.1)
plt.plot(testx[:, 0], predict[:, 0], c="k", lw=2, label="拟合曲线")
plt.legend()
plt.xlabel("x")
plt.ylabel("d")
plt.show() 

    