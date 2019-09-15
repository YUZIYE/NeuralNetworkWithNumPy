import tensorflow as tf 

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() 


import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.gridspec import GridSpec 

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

gs = GridSpec(6, 8, wspace=0.1, hspace=0.1)
fig = plt.figure(constrained_layout=True) 

images = np.reshape(train_images, [-1, 28, 28]) 

for itx in range(6):
    for ity in range(8):
        ax = fig.add_subplot(gs[itx, ity])
        ax.imshow(images[itx*80+ity], cmap=plt.get_cmap("Greys"))
        ax.text(3, 6, "%d"%train_labels[itx*80+ity])
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        #ax.legend()
plt.show()

ax = fig.add_subplot(gs[0, 2:])



