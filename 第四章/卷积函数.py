import numpy as np 
def conv2d(inputs, filters, par, stride=1, padding="same"):
    """
    二维卷积函数
    inputs:输入图像
    filters:卷积核心
    stride:步长（长宽步长一致）
    padding:边缘填充策略
    """
    B, H, W, C = np.shape(inputs)
    K, K, C, C2 = np.shape(filters)
    if padding == "same":
        H2 = int((H-0.1)//stride + 1)
        W2 = int((W-0.1)//stride + 1)
        # 边缘填充
        pad_h_2 = K + (H2 - 1) * stride - H
        pad_w_2 = K + (W2 - 1) * stride - W
        pad_h_left = int(pad_h_2//2)
        pad_h_right = int(pad_h_2 - pad_h_left)
        pad_w_left = int(pad_w_2//2)
        pad_w_right = int(pad_w_2 - pad_w_left)
        X = np.pad(inputs, ((0, 0), 
                            (pad_h_left, pad_h_right),
                            (pad_w_left, pad_w_right), 
                            (0, 0)), 
                            'constant', 
                            constant_values=0)
    elif padding == "valid":
        H2 = int((H - K)//stride + 1)
        W2 = int((W - K)//stride + 1)
        X = inputs
    else:
        raise "parameter error"
    out = np.zeros([B, H2, W2, C2])
    for itr1 in range(B):
        for itr2 in range(H2):
            for itr3 in range(W2):
                for itrc in range(C2):
                    itrh = itr2 * stride
                    itrw = itr3 * stride
                    out[itr1, itr2, itr3, itrc] = \
                        np.sum(X[itr1, itrh:itrh+K, itrw:itrw+K, :]\
                             * filters[:,:,:,itrc])
        return out