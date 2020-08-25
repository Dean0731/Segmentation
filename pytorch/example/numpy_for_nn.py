# @Time     : 2020/8/25 15:11
# @File     : numpy_for_nn
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release

import torch
import numpy as np
N,D_in,H,D_out = 64,1000,100,10

x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)

w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)


learning_rate = 1e-6
for t in range(500):
    # 前向传播
    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)

    # 计算损失，需要损失函数

    loss = np.square(y_pred-y).sum() # 均方误差
    print(t,loss)

    # 反向传播，目的是要算出 d(loss)/d(w1)  = dloss/dpred * dpred/dw2 * dw2
    # 1,将损失函数看做 y = (pred-true)^2
    grad_y_pred = 2.0*(y_pred-y)
    # 2,矩阵求导 Y=AX, dy/dx = 矩阵A的转置，Y=XA  dy/dx = 矩阵A
    grad_w2 = h_relu.T.dot(grad_y_pred)

    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 =  x.T.dot(grad_h)


    # 更新梯度
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2


