import numpy as np 

# 生成数据
x = np.random.normal(-1, 0.3, [2000, 2])
x[500:1000] = np.random.normal(1, 0.3, [500, 2])
x[1000:1500] = np.random.normal(0, 0.3, [500, 2]) + np.array([-1, 1])
x[1500:2000] = np.random.normal(0, 0.3, [500, 2]) + np.array([1, -1])
d = np.zeros([2000])
d[1000:] = 1


import tensorflow as tf 
inputx = tf.placeholder(tf.float32, [None, 2])
inputy = tf.placeholder(tf.int32, [None]) 
input_onehot = tf.one_hot(inputy, 2)
h = tf.layers.dense(inputx, 2, activation=tf.nn.sigmoid) 
logits = tf.layers.dense(h, 2, activation=None)  
loss = tf.losses.softmax_cross_entropy(input_onehot, logits)
prob = tf.nn.softmax(logits)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

#定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
#定义log输出位置，用于tensorboard观察。
train_writer = tf.summary.FileWriter("model/xor", sess.graph)
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('model'))
for itr in range(1):
    idx = np.random.randint(0, len(x), 64)
    inx = x[idx] 
    ind = d[idx] 
    sess.run(train_step, feed_dict={inputx: inx, inputy: ind})
    if itr % 10 == 0:
        print(itr)
        saver.save(sess, "model/xor", global_step=itr)

import matplotlib.pyplot as plt 
# 测试绘图 
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

#plt.subplot(211)
plt.scatter(x[:1000, 0], x[:1000, 1], c='k', marker="$1$", label="第一类样本", alpha=0.5)
plt.scatter(x[1000:, 0], x[1000:, 1], c='k', marker="$2$", label="第二类样本") 
plt.legend()
plt.axis("equal")
plt.xlabel(u"$x_1$")
plt.ylabel(u"$x_2$")
"""
plt.subplot(212)



plt.scatter(x[:1000, 0], x[:1000, 1], c='k', marker="$1$", label="第一类样本", alpha=0.5)
plt.scatter(x[1000:, 0], x[1000:, 1], c='k', marker="$2$", label="第二类样本") 

x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
x1, x2 = np.meshgrid(x1, x2)
x1 = np.reshape(x1, [-1, 1])
x2 = np.reshape(x2, [-1, 1])
X = np.concatenate([x1, x2], axis=1) 
Y = sess.run(prob, feed_dict={inputx:X}) 
x1 = np.reshape(x1, [200, 200])
x2 = np.reshape(x2, [200, 200])
x3 = np.reshape(Y[:, 0], [200, 200])

C = plt.contour(x1, x2, x3, 8, colors='black', label="多层全连接网络预测属于第一类概率")
plt.clabel(C,inline=1, fontsize=10, label="多层全连接网络预测属于第一类概率")

plt.legend()
plt.axis("equal")
plt.xlabel(u"$x_1$")
plt.ylabel(u"$x_1$")
"""
plt.show() 
