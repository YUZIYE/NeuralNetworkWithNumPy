#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
rnn 网络实现
=====================
"""

import numpy as np
np.random.seed(0)

import numpy as np

class NN():
    def __init__(self):
        """
        定义可训练参数
        """
        self.value = []
        self.d_value = []
        self.outputs = []
        self.layer = []
        self.layer_name = []
    def tanh(self, x, n_layer=None, layer_par=None):
        epx = np.exp(x)
        enx = np.exp(-x)
        return (epx-enx)/(epx+enx)
    def d_tanh(self, x, n_layer=None, layer_par=None):
        e2x = np.exp(2 * x)
        return 4 * e2x / (1 + e2x) ** 2
    def _rnncell(self, X, n_layer, layer_par):
        """
        RNN正向传播层
        """
        W = self.value[n_layer][0]
        bias = self.value[n_layer][1]
        b, h, c = np.shape(X)
        _, h2 = np.shape(W)
        outs = []
        stats = []
        s = np.zeros([b, h2])
        for itr in range(h):
            x = X[:, itr, :]
            stats.append(s)
            inx = np.concatenate([x, s], axis=1)
            out = np.dot(inx, W) + bias
            out = self.tanh(out)
            s = out
            outs.append(out)
        outs = np.transpose(outs, (1, 0, 2))
        stats= np.transpose(stats, (1, 0, 2))
        return [outs, stats]
    def _d_rnncell(self, error, n_layer, layer_par):
        """
        BPTT层，此层使用上一层产生的Error产生向前一层传播的error
        """
        inputs = self.outputs[n_layer][0]
        states = self.outputs[n_layer + 1][1]
        b, h, insize = np.shape(inputs)
        back_error = [np.zeros([b, insize]) for itr in range(h)]
        W = self.value[n_layer][0]
        bias = self.value[n_layer][1]
        dw = np.zeros_like(W)
        db = np.zeros_like(bias)
        w1 = W[:insize, :]
        w2 = W[insize:, :]
        for itrs in range(h - 1, -1, -1):
            # 每一个时间步都要进行误差传播
            if len(error[itrs]) == 0:
                continue
            else:
                err = error[itrs]
            for itr in range(itrs, -1, -1):
                h = states[:, itr, :]
                x = inputs[:, itr, :]
                inx = np.concatenate([x, h], axis=1)
                h1 = np.dot(inx, W) + bias
                d_fe = self.d_tanh(h1)
                err = d_fe * err
                # 计算可训练参数导数
                dw[:insize, :] += np.dot(x.T, err)
                dw[insize:, :] += np.dot(h.T, err)
                db += np.sum(err, axis=0)
                # 计算传递误差
                back_error[itr] += np.dot(err, w1.T)
                err = np.dot(err, w2.T)
        self.d_value[n_layer][0] = dw
        self.d_value[n_layer][1] = db
        return back_error
    def _embedding(self, inputs, n_layer, layer_par):
        W = self.value[n_layer][0]
        F, E = np.shape(W)
        B, L = np.shape(inputs)
        # 转换成one-hot向量
        inx = np.zeros([B * L, F])
        inx[np.arange(B * L), inputs.reshape(-1)] = 1
        inx = inx.reshape([B, L, F])
        # 乘以降维矩阵
        embed = np.dot(inx, W)
        return [embed]
    def _d_embedding(self, in_error, n_layer, layer_par):
        inputs = self.outputs[n_layer][0]
        W = self.value[n_layer][0]
        F, E = np.shape(W)
        B, L = np.shape(inputs)
        inx = np.zeros([B * L, F])
        inx[np.arange(B * L), inputs.reshape(-1)] = 1
        error = np.transpose(in_error, (1, 0, 2))
        _, _, C = np.shape(error)
        error = error.reshape([-1, C])
        # 计算降维矩阵的导数
        self.d_value[n_layer][0] = np.dot(inx.T, error)
        return []
    def _text_error(self, inputs, n_layer, layer_par):
        return [inputs[:, -1, :]]
    def _d_text_error(self, in_error, n_layer, layer_par):
        X = self.outputs[n_layer][0]
        b, h, c = np.shape(X)
        error = [[] for itr in range(h)]
        error[-1] = in_error
        return error
    def _matmul(self, inputs, n_layer, layer_par):
        W = self.value[n_layer][0]
        return [np.dot(inputs, W)]
    def _d_matmul(self, in_error, n_layer, layer_par):
        W = self.value[n_layer][0]
        inputs = self.outputs[n_layer][0]
        self.d_value[n_layer][0] = np.dot(inputs.T, in_error)
        error = np.dot(in_error, W.T)
        return error
    def _bias_add(self, inputs, n_layer, layer_par):
        b = self.value[n_layer][0]
        return [inputs + b]
    def _d_bias_add(self, in_error, n_layer, layer_par):
        self.d_value[n_layer][0] = np.sum(in_error, axis=0)
        return in_error
    def _sigmoid(self, X, n_layer=None, layer_par=None):
        return [1/(1+np.exp(-X))]
    def _d_sigmoid(self, in_error, n_layer=None, layer_par=None):
        X = self.outputs[n_layer][0]
        #print("teterror", n_layer, in_error,  np.exp(-X)/(1 + np.exp(-X)) ** 2)
        return in_error * np.exp(-X)/(1 + np.exp(-X)) ** 2
    def _loss_square(self, Y, n_layer, layer_par):
        B = np.shape(Y)[0]
        return [np.square(self.outputs[-1] - Y)/B]
    def _d_loss_square(self, Y, n_layer, layer_par):
        B = np.shape(Y)[0]
        inputs = self.outputs[-2][0]
        return 2 * (inputs - Y)
    def basic_rnn(self, w, b):
        self.value.append([w, b])
        self.d_value.append([np.zeros_like(w), np.zeros_like(b)])
        self.layer.append((self._rnncell, None, self._d_rnncell, None))
        self.layer_name.append("rnn")
    def bias_add(self, bias):
        self.value.append([bias])
        self.d_value.append([np.zeros_like(bias)])
        self.layer.append((self._bias_add, None, self._d_bias_add, None))
        self.layer_name.append("bias_add")
    def matmul(self, filters):
        self.value.append([filters])
        self.d_value.append([np.zeros_like(filters)])
        self.layer.append((self._matmul, None, self._d_matmul, None))
        self.layer_name.append("matmul")
    def embedding(self, w):
        self.value.append([w])
        self.d_value.append([np.zeros_like(w)])
        self.layer.append((self._embedding, None, self._d_embedding, None))
        self.layer_name.append("embedding")
    def text_error(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._text_error, None, self._d_text_error, None))
        self.layer_name.append("text_error")
    def loss_square(self):
        self.value.append([])
        #print(filters)
        self.d_value.append([])
        self.layer.append((self._loss_square, None, self._d_loss_square, None))    
        self.layer_name.append("loss")       
    def sigmoid(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._sigmoid, None, self._d_sigmoid, None))
        self.layer_name.append("sigmoid")        
    def forward(self, X):
        self.outputs.append([X])
        net = [X]
        for idx, lay in enumerate(self.layer):
            method, layer_par, _, _ = lay
            net = method(net[0], idx, layer_par)
            self.outputs.append(net)
        return self.outputs[-2][0]
    def backward(self, Y):
        error = self.layer[-1][2](Y, None, None)
        self.n_layer = len(self.value)
        for itr in range(self.n_layer-2, -1, -1):
            _, _, method, layer_par = self.layer[itr]
            #print("++++-", np.shape(error), np.shape(Y))
            error = method(error, itr, layer_par)
        return error
    def apply_gradient(self, eta):
        for idx, itr in enumerate(self.d_value):
            if len(itr) == 0: continue
            for idy, val in enumerate(itr):
                self.value[idx][idy] -= val * eta / 20
    def fit(self, X, Y):
        self.forward(X)
        self.backward(Y)
        self.apply_gradient(0.1)
    def predict(self, X):
        self.forward(X)
        return self.outputs[-2]
    def _relu(self, X, *args, **kw):
        return [(X + np.abs(X))/2.]
    def _d_relu(self, in_error, n_layer, layer_par):
        X = self.outputs[n_layer][0]
        drelu = np.zeros_like(X)
        drelu[X>0] = 1
        return in_error * drelu
    def relu(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._relu, None, self._d_relu, None))
        self.layer_name.append("relu")

def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id
def read_category():
    """读取分类目录，固定"""
    categories = ['pos', 'neg']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
categories, cat_to_id = read_category()
words, word_to_id = read_vocab("data/vocab.txt")
x_pad, y_pad = process_file("data/emotion.txt", word_to_id, cat_to_id, max_length=36)
vocab_size = len(words)

method = NN()

files = np.load("par.npz")
names = files['name']
datas = files['data']

ew = datas[0]
w1 = datas[1]
b1 = datas[2]
w2 = datas[3]
b2 = datas[4]
w3 = datas[5]
b3 = datas[6]
w4 = datas[7]
b4 = datas[8]

method.embedding(ew)
method.basic_rnn(w1, b1)
method.basic_rnn(w2, b2)
method.text_error()
method.matmul(w3)
method.bias_add(b3)
method.relu()
method.matmul(w4)
method.bias_add(b4)
method.sigmoid()
method.loss_square()

N = len(x_pad)
for itr in range(1000):
    idx = np.random.randint(0, N, [20])
    inx = x_pad[idx]
    iny = y_pad[idx]
    pred = method.forward(inx)
    method.backward(iny)
    prd1 = np.argmax(pred, axis=1)
    prd2 = np.argmax(iny, axis=1)
    #print(pred)
    #print("temp", np.sum(prd1==prd2)/len(idx))
    method.apply_gradient(0.001)
    if itr% 20 == 0:
        idx = np.random.randint(0, N, [400])
        inx = x_pad[idx]
        iny = y_pad[idx]
        pred = method.forward(inx)
        prd1 = np.argmax(pred, axis=1)
        prd2 = np.argmax(iny, axis=1)
        print(np.sum(prd1==prd2)/len(idx))
