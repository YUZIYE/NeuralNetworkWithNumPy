import numpy as np

class NN():
    def __init__(self):
        # 用于保存可训练参数
        self.value = []
        # 用于保存可训练参数的导数
        self.d_value = []
        # 用于保留每一层的输出
        self.outputs = []
        # 用于保存网络结构
        self.layer = []
        # 保存网络名称
        self.layer_name = []
    def _conv2d(self, inputs, filters, par):
        """
        卷积网络正向传播过程
        inputs:输入图像
        filters:卷积核心
        par:中包含步长stride和padding
        """
        stride, padding = par
        # 获取矩阵shape
        B, H, W, C = np.shape(inputs)
        K, K, C, C2 = np.shape(filters)
        if padding == "SAME":
            # SAME边界条件时需要对图像边界进行补0
            H2 = np.ceil(H/stride + 1)
            W2 = np.ceil(W/stride + 1)
            pad_h_2 = K + (H2 - 1) * stride - H
            pad_w_2 = K + (W2 - 1) * stride - W
            pad_h_left = int(pad_h_2//2)
            pad_h_right = int(pad_h_2 - pad_h_left)
            pad_w_left = int(pad_w_2//2)
            pad_w_right = int(pad_w_2 - pad_w_left)
            X = np.pad(inputs, 
                        ((0, 0), 
                         (pad_h_left, pad_h_right),
                         (pad_w_left, pad_w_right), 
                         (0, 0)), 
                        'constant', 
                        constant_values=0)
        elif padding == "VALID":
            # VALID边界条件仅计算有值位置
            H2 = int((H - K)//stride + 1)
            W2 = int((W - K)//stride + 1)
            X = inputs
        # 定义输出
        out = np.zeros([B, H2, W2, C2])
        for itr1 in range(B):
            for itr2 in range(H2):
                for itr3 in range(W2):
                    for itrc in range(C2):
                        itrh = itr2 * stride
                        itrw = itr3 * stride
                        out[itr1, itr2, itr3, itrc] = np.sum(X[itr1, itrh:itrh+K, itrw:itrw+K, :] * filters[:,:,:,itrc])
        return out
    def _d_conv2d(self, in_error, n_layer, layer_par=None):
        """
        卷积层反向传播
        in_error:上一层计算误差
        n_layer:层数
        """
        # 获取正向计算时信息
        stride, padding = self.layer[n_layer][1]
        inputs = self.outputs[n_layer]
        filters = self.value[n_layer]
        B, H, W, C = np.shape(inputs)
        K, K, C, C2 = np.shape(filters)

        if padding == "SAME":
            H2 = np.ceil(H/stride + 1)
            W2 = np.ceil(W/stride + 1)
            pad_h_2 = K + (H2 - 1) * stride - H
            pad_w_2 = K + (W2 - 1) * stride - W
            pad_h_left = int(pad_h_2//2)
            pad_h_right = int(pad_h_2 - pad_h_left)
            pad_w_left = int(pad_w_2//2)
            pad_w_right = int(pad_w_2 - pad_w_left)
            X = np.pad(inputs, ((0, 0), 
                                (pad_h_left, pad_h_right),
                                (pad_w_left, pad_w_right), 
                                (0, 0)), 'constant', constant_values=0)
        elif padding == "VALID":
            H2 = int((H - K)//stride + 1)
            W2 = int((W - K)//stride + 1)
            X = inputs
        # 计算本层可训练参数的导数
        error = np.zeros_like(X)
        for itr1 in range(B):
            for itr2 in range(H2):
                for itr3 in range(W2):
                    for itrc in range(C2):
                        itrh = itr2 * stride
                        itrw = itr3 * stride
                        error[itr1, itrh:itrh+K, itrw:itrw+K, :] += in_error[itr1, itr2, itr3, itrc] *  filters[:,:,:,itrc]
        self.d_value[n_layer] = np.zeros_like(self.value[n_layer])
        # 计算反向传播误差
        for itr1 in range(B):
            for itr2 in range(H2):
                for itr3 in range(W2):
                    for itrc in range(C2):
                        itrh = itr2 * stride
                        itrw = itr3 * stride
                        self.d_value[n_layer][:, :, :, itrc] += in_error[itr1, itr2, itr3, itrc] * X[itr1, itrh:itrh+K, itrw:itrw+K, :]
        return error[:, pad_h_left:-pad_h_right, pad_w_left:-pad_w_right, :]
    def _flatten(self, X, *args, **kw):
        B = np.shape(X)[0]
        return np.reshape(X, [B, -1])
    def _d_flatten(self, in_error, n_layer, layer_par):
        shape = np.shape(self.outputs[n_layer])
        return np.reshape(in_error, shape)
    def flatten(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._flatten, None, self._d_flatten, None))
        self.layer_name.append("flatten")
    def _matmul(self, inputs, W, *ag, **kw):
        """
        矩阵乘法正向传播过程
        inputs:本层输入
        W:本层可训练参数
        """
        return np.dot(inputs, W)
    def _d_matmul(self, in_error, n_layer, layer_par):
        """
        矩阵乘法反向传播过程
        in_error:上一层误差
        n_layer:层数
        """
        # 本层可训练参数
        W = self.value[n_layer]
        # 获取正向传播输入
        inputs = self.outputs[n_layer]
        # 计算可训练参数导数
        self.d_value[n_layer] = np.dot(inputs.T, in_error)
        # 计算反向传播误差
        error = np.dot(in_error, W.T)
        return error
    def _bias_add(self, inputs, b, *args, **kw):
        return inputs + b
    def _d_bias_add(self, in_error, n_layer, *args, **kw):
        shape = np.shape(in_error)
        dv = []
        if len(shape) == 2:
            self.d_value[n_layer] = np.sum(in_error, axis=0)
        else:
            dv = np.array(
                [np.sum(in_error[:, :, :, itr]) for itr in range(shape[-1])])
            self.d_value[n_layer] = np.squeeze(np.array(dv))
        return in_error
    def _maxpool(self, X, _, stride, *args, **kw):
        B, H, W, C = np.shape(X)
        X_new = np.reshape(X, [B, H//stride, stride, W//stride, stride, C])
        return np.max(X_new, axis=(2, 4))
    def _d_maxpool(self, in_error, n_layer, layer_par):
        stride = layer_par
        X = self.outputs[n_layer]
        Y = self.outputs[n_layer + 1]
        expand_y = np.repeat(np.repeat(Y, stride, axis=1), stride, axis=2)
        expand_e = np.repeat(np.repeat(in_error, stride, axis=1), stride, axis=2)
        return expand_e * (expand_y == X)
    def _sigmoid(self, X, *args, **kw):
        return 1/(1+np.exp(-X))
    def _d_sigmoid(self, in_error, n_layer, layer_par):
        X = self.outputs[n_layer]
        return in_error * np.exp(-X)/(1 + np.exp(-X)) ** 2
    def _relu(self, X, *args, **kw):
        return (X + np.abs(X))/2.
    def _d_relu(self, in_error, n_layer, layer_par):
        X = self.outputs[n_layer]
        drelu = np.zeros_like(X)
        drelu[X>0] = 1
        return in_error * drelu
    def _loss_square(self, Y, *args, **kw):
        B = np.shape(Y)[0]
        return np.square(self.outputs[-2] - Y)
    def _d_loss_square(self, Y, *args, **kw):
        B = np.shape(Y)[0]
        return 2 * (self.outputs[-2] - Y)
    def conv2d(self, filters, stride, padding="SAME"):
        self.value.append(filters)
        self.d_value.append(np.zeros_like(filters))
        self.layer.append((self._conv2d, (stride, padding), self._d_conv2d, None))
        self.layer_name.append("conv2d")
    def bias_add(self, bias, *args, **kw):
        self.value.append(bias)
        self.d_value.append(np.zeros_like(bias))
        self.layer.append((self._bias_add, None, self._d_bias_add, None))
        self.layer_name.append("bias_add")
    def matmul(self, filters, *args, **kw):
        self.value.append(filters)
        self.d_value.append(np.zeros_like(filters))
        self.layer.append((self._matmul, None, self._d_matmul, None))
        self.layer_name.append("matmul")
    def maxpool(self, stride, *args, **kw):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._maxpool, stride, self._d_maxpool, stride))
        self.layer_name.append("maxpool")
    def loss_square(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._loss_square, None, self._d_loss_square, None))    
        self.layer_name.append("loss")       
    def sigmoid(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._sigmoid, None, self._d_sigmoid, None))
        self.layer_name.append("sigmoid")
    def relu(self):
        self.value.append([])
        self.d_value.append([])
        self.layer.append((self._relu, None, self._d_relu, None))
        self.layer_name.append("relu")     
    def forward(self, X):
        self.outputs = []
        self.outputs.append(X)
        net = X
        for idx, lay in enumerate(self.layer):
            method, layer_par, _, _ = lay
            net = method(net, self.value[idx], layer_par)
            self.outputs.append(net)
        return
    def backward(self, Y):
        error = self.layer[-1][2](Y)
        self.n_layer = len(self.value)
        for itr in range(self.n_layer-2, -1, -1):
            _, _, method, layer_par = self.layer[itr]
            error = method(error, itr, layer_par)
    def apply_gradient(self, eta):
        for idx, itr in enumerate(self.d_value):
            if len(itr) == 0: continue
            self.value[idx] -= itr * eta
    def fit(self, X, Y):
        self.forward(X)
        self.backward(Y)
        self.apply_gradient(0.01)
    def compute_gradients(self, X, Y):
        self.forward(X)
        self.backward(Y)
        #self.apply_gradient(0.1)
    def predict(self, X):
        self.forward(X)
        return self.outputs[-2]
    def model(self):
        for idx, itr in enumerate(self.layer_name):
            print("Layer %d: %s"%(idx, itr))
    

mtd = NN()

cw1 = np.random.uniform(-0.1, 0.1, [5, 5, 1, 32])
cb1 = np.zeros([32])
cw2 = np.random.uniform(-0.1, 0.1, [5, 5, 6, 16])
cb2 = np.zeros([16])
cw3 = np.random.uniform(-0.1, 0.1, [4, 4, 6, 16])
cb3 = np.zeros([16])

fw1 = np.random.uniform(-0.1, 0.1, [7 * 7 * 32, 10])
fb1 = np.zeros([10])
fw2 = np.random.uniform(-0.1, 0.1, [120, 84])
fb2 = np.zeros([84])
fw3 = np.random.uniform(-0.1, 0.1, [84, 10])
fb3 = np.zeros([10])

mtd.conv2d(cw1, 1)
mtd.bias_add(cb1)
mtd.relu()
mtd.maxpool(4)

mtd.flatten()

mtd.matmul(fw1)
mtd.bias_add(fb1)
mtd.sigmoid()


mtd.loss_square()
mtd.model()
train = np.load("train.npz")
test = np.load("test.npz")
train_image = np.reshape(train['images'], [-1, 28, 28, 1])
test_image = np.reshape(test['images'], [-1, 28, 28, 1])
train_label = train['labels']
test_label = test['labels']

mean = np.mean(train_image)
std = np.std(train_image)
for itr in range(600):
    idx = np.random.randint(0, 20000, 10)
    inx = (train_image[idx])
    iny = train_label[idx]
    mtd.fit(inx, iny)
    if itr % 5 == 0:
        idx = np.random.randint(0, 20000, 200)
        inx = train_image[idx]
        iny = train_label[idx]
        pred2 = mtd.predict(inx)
        error2 = np.sum(np.argmax(pred2, axis=1) == np.argmax(iny, axis=1))/len(iny)
        print(itr, error2)

