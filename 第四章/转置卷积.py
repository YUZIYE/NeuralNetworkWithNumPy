import numpy as np 
import tensorflow as tf  

kernel = tf.ones([1, 1, 1, 1])
images = tf.ones([1, 2, 2, 1]) 
delit = tf.nn.conv2d_transpose(images, kernel, [1, 4, 4, 1], strides=[1, 2, 2, 1])

print(delit.shape)
sess = tf.Session() 
out = sess.run(delit)
print(out[0, :, :, 0])