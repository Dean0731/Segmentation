# @Time     : 2020/8/25 15:11
# @File     : pytorch_for_nn
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import torch
N,D_in,H,D_out = 64,1000,100,10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

w1 = torch.randn(D_in,H,requires_grad=True)  # 是否需要求梯度，默认不求梯度
w2 = torch.randn(H,D_out,requires_grad=True)


learning_rate = 1e-6
for t in range(200):
    # 前向传播
    h = x.mm(w1)
    h_relu = h.clamp(min=0) # 将负数变为0
    y_pred = h_relu.mm(w2)

    # 计算损失，需要损失函数

    loss = (y_pred-y).pow(2).sum() # 均方误差函数，数学上是公式，在这里是计算图，这样才能求导数

    print(t,loss.item())

    # 反向传播，目的是要算出导数 d(loss)/d(w1)  = dloss/dpred * dpred/dw2 * dw2
    # 矩阵求导 Y=AX, dy/dx = 矩阵A的转置，Y=XA  dy/dx = 矩阵A
    # 将损失函数看做 y = (pred-true)^2
    if False:
        grad_y_pred = 2.0*(y_pred-y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h<0] = 0
        grad_w1 =  x.t().mm(grad_h)
        # 更新梯度
        w1 -= learning_rate*grad_w1
        w2 -= learning_rate*grad_w2
    else:
        loss.backward()
        with torch.no_grad(): # 当做普通计算，不用计算图，节省内存
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()



