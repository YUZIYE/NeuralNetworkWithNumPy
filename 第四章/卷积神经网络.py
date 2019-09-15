
# by cangye@hotmail.com
# TensorFlow入门实例

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import pyopencl as cl
mnist = input_data.read_data_sets("data/", one_hot=True)

#卷积函数
def conv2d_layer(input_tensor, size=1, feature=128, name='conv1d'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel', 
                                  (size, size, shape[-1], feature), 
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))#初始化值很重要，不好的初始化值比如文章中的初始化值会使得迭代收敛极为缓慢。
        b = tf.get_variable('b', [feature], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 2, 2, 1], padding='VALID') + b
    return tf.nn.relu(out), kernel, b
#全链接函数
def full_layer(input_tensor, out_dim, name='full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return out, W, b


with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")
x2d = tf.placeholder(tf.float32, [None,28,28,1])
net1, w1, b1 = conv2d_layer(x2d, size=4, feature=64, name='conv1')

#print(net.get_shape().as_list())
net2, w2, b2 = conv2d_layer(net1, size=4, feature=64, name='conv2')
#print(net.get_shape().as_list())
net3, w3, b3 = conv2d_layer(net2, size=4, feature=64, name='conv3')
#print(net.get_shape().as_list())
#flatten层，用于将三维的图形数据展开成一维数据，用于全链接层
net4 = tf.contrib.layers.flatten(net3)
y, w4, b4=full_layer(net4, 10, name='full')

with tf.variable_scope("loss"):
    #定义loss函数
    ce=tf.square(label-tf.nn.sigmoid(y))
    loss = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#用于训练参数的保存
saver = tf.train.Saver()
#载入保存的权值
saver.restore(sess, tf.train.latest_checkpoint('model'))

for itr in range(0):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x2d: np.reshape(batch_xs, [-1, 28, 28, 1]), label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x2d: np.reshape(mnist.test.images, [-1, 28, 28, 1]),
                                        label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model','handwriting'), global_step=itr)

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return (np.abs(x) + x)/2
class cl_kernel():
    def __init__(self,device=0):
        platforms  = cl.get_platforms()  
        self.ctx = cl.Context(dev_type=cl.device_type.ALL,  
                            properties=[(cl.context_properties.PLATFORM,platforms[0])])  
        #self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.prg = cl.Program(self.ctx, """
                        __kernel void conv2d(
                            __global const float* x, 
                            __global const float* b, 
                            __global const float* kr, 
                            __global float* r, 
                            int h,
                            int w,
                            int c,
                            int h2,
                            int w2,
                            int c2,
                            int ks,
                            int s)
                        {
                        int itr2 = get_global_id(0);
                        int itr3 = get_global_id(1);
                        int itr4 = get_global_id(2);
                        float temp=0;
                        int i,j,k;
                        int itrh = itr2 * s;
                        int itrw = itr3 * s;
                        for(i=0;i<ks;i++){
                            for(j=0;j<ks;j++){
                                for(k=0;k<c;k++)
                                    temp+=x[(itrh+i)*w*c + (itrw+j)*c + k] * kr[i*ks*c*c2+j*c*c2+k*c2+itr4];
                                }
                            }
                        r[(itr2)*w2*c2 + (itr3)*c2 + itr4]=temp + b[itr4];
                        }
                        __kernel void matmul(
                                    __global const float* x, 
                                    __global const float* w, 
                                    __global const float* b,
                                    __global float* ret,
                                    const int Ndim,
                                    const int Mdim,
                                    const int Pdim)
                                {
                                    int i = get_global_id(0);
                                    int j = get_global_id(1);

                                    int k;
                                    float tmp;

                                    if ((i < Ndim) && (j < Mdim)) {
                                        tmp = 0.0;
                                        for (k = 0; k < Pdim; k++)
                                            tmp += x[i*Pdim + k] * w[k*Mdim + j];
                                        ret[i*Mdim + j] = tmp+b[j];
                                    }
                                }
                        """).build()
    def conv2d(self, x, kernel, bias, s=2):
        k1, k2, c1, c2 = np.shape(kernel)
        b, h, w, c = np.shape(x)
        h2 = int((h-k1)/s)+1
        w2 = int((w-k2)/s)+1
        b_in = np.reshape(bias, [-1]).astype(np.float32)
        k_in = np.reshape(kernel, [-1]).astype(np.float32)
        b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_in)
        k_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=k_in)
        ret = np.ones([b, h2, w2, c2])
        out = np.zeros([h2, w2, c2]).astype(np.float32)
        res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
        for idx, itr in enumerate(x):
            x_in = np.reshape(itr, [-1]).astype(np.float32)
            mf = cl.mem_flags
            a_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=x_in)
            self.prg.conv2d(self.queue, (h2, w2, c2), None, a_g, 
                                                                b_g, 
                                                                k_g, 
                                                                res_g, 
                                                                #np.array(b).astype(np.int32),
                                                                np.array(h).astype(np.int32),
                                                                np.array(w).astype(np.int32),
                                                                np.array(c).astype(np.int32),
                                                                np.array(h2).astype(np.int32),
                                                                np.array(w2).astype(np.int32),
                                                                np.array(c2).astype(np.int32),
                                                                np.array(k2).astype(np.int32),
                                                                np.array(s).astype(np.int32))
            cl.enqueue_copy(self.queue, out, res_g)
            ret[idx]=out
        return relu(ret)
    def full(self, x, w, b):
        sp = np.shape(x)
        if len(sp)==4:
            x = np.reshape(x, [sp[0], -1])
        N, P = np.shape(x)
        P, M = np.shape(w)
        sp = np.shape(x)
        x_in = np.reshape(x, [-1]).astype(np.float32)
        w_in = np.reshape(w, [-1]).astype(np.float32)
        b_in = np.reshape(b, [-1]).astype(np.float32)
        x_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=x_in)
        w_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=w_in)
        b_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=b_in)
        out = np.zeros([N, M]).astype(np.float32)
        res_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
        self.prg.matmul(self.queue, (N, M), None, x_g, 
                                                    w_g, 
                                                    b_g, 
                                                    res_g, 
                                                    np.array(N).astype(np.int32),
                                                    np.array(M).astype(np.int32),
                                                    np.array(P).astype(np.int32))
        cl.enqueue_copy(self.queue, out, res_g)
        return out
def conv2d(x, kernel, bias, s=2):
    k1, k2, c1, c2 = np.shape(kernel)
    b, h, w, c = np.shape(x)
    h2 = int((h-k1)/s)+1
    w2 = int((w-k2)/s)+1
    out = np.zeros([b, h2, w2, c2])
    for itr1 in range(b):
        for itr2 in range(h2):
            for itr3 in range(w2):
                for itrc in range(c2):
                    itrh = itr2 * s
                    itrw = itr3 * s
                    out[itr1, itr2, itr3, itrc] = relu(np.sum(x[itr1, itrh:itrh+k1, itrw:itrw+k2, :] * kernel[:,:,:,itrc]) + bias[itrc])
    return out
def full(x, kernel, bias):
    sp = np.shape(x)
    if len(sp)==4:
        x = np.reshape(x, [sp[0], -1])
    return np.dot(x, kernel) + bias

nw1, nb1, nw2, nb2, nw3, nb3, nw4, nb4 = sess.run([w1.value(), b1.value()
                     , w2.value(), b2.value()
                     , w3.value(), b3.value()
                     , w4.value(), b4.value()
                     ])
import time


st = time.clock()
clk = cl_kernel(0)
inputs = np.reshape(mnist.test.images[:100,:], [-1, 28, 28, 1])
#inputs = np.ones([1, 4, 4, 2])
net = clk.conv2d(inputs, nw1, nb1)
net = clk.conv2d(net, nw2, nb2)
net = clk.conv2d(net, nw3, nb3)
f_s = time.clock()
ya = clk.full(net, nw4, nb4)
f_e = time.clock()
a1 = np.argmax(ya, axis=1)
a2 = np.argmax(mnist.test.labels[:100,:], axis=1)
print(np.sum(a1==a2)/len(ya))
ed = time.clock()
print("GPU CL time:%f, full_connect:%s"%(ed-st, f_e-f_s))

st = time.clock()
clk = cl_kernel(1)
inputs = np.reshape(mnist.test.images[:100,:], [-1, 28, 28, 1])
#inputs = np.ones([1, 4, 4, 2])
net = clk.conv2d(inputs, nw1, nb1)
net = clk.conv2d(net, nw2, nb2)
net = clk.conv2d(net, nw3, nb3)
f_s = time.clock()
ya = clk.full(net, nw4, nb4)
f_e = time.clock()
a1 = np.argmax(ya, axis=1)
a2 = np.argmax(mnist.test.labels[:100,:], axis=1)
print(np.sum(a1==a2)/len(ya))
ed = time.clock()
print("CPU CL time:%f, full_connect:%s"%(ed-st, f_e-f_s))



st = time.clock()
inputs = np.reshape(mnist.test.images[:100,:], [-1, 28, 28, 1])
#inputs = np.ones([1, 4, 4, 2])
net = conv2d(inputs, nw1, nb1)
net = conv2d(net, nw2, nb2)
net = conv2d(net, nw3, nb3)
f_s = time.clock()
ya = full(net, nw4, nb4)
f_e = time.clock()
a1 = np.argmax(ya, axis=1)
a2 = np.argmax(mnist.test.labels[:100,:], axis=1)
print(np.sum(a1==a2)/len(ya))
ed = time.clock()
print("Numpy time:%f, full_connect:%s"%(ed-st, f_e-f_s))