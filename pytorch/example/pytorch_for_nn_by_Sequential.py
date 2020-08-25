# @Time     : 2020/8/25 16:39
# @File     : pytorch_for_nn_by_Sequential
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import torch
import torch.nn as nn
N,D_in,H,D_out = 64,1000,100,10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

model = nn.Sequential(
    nn.Linear(D_in,H,bias=False),
    nn.ReLU(),
    nn.Linear(H,D_out,bias=False),
)
# nn.init.normal_(model[0].weight) # 初始化为正态分布，节省训练步骤
# nn.init.normal_(model[2].weight)

loss_fn = nn.MSELoss(reduction='sum')

# learning_rate 不同优化算法不同 adam适合e-3~e-4，sgd适合 e-6
# 不同的优化算法，weight初始化也不相同，sgd初始化为正态分布效果好，adam则不同，随机就行

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #

for t in range(500):
    # 前向传播
    y_pred = model(x)
    # 计算损失，需要损失函数
    loss = loss_fn(y_pred,y) # 均方误差函数，数学上是公式，在这里是计算图，这样才能求导数
    print(t,loss.item())
    # 反向传播,求梯度值
    loss.backward() # 将梯度值放在参数中
    if False:
        with torch.no_grad(): # 当做普通计算，不用计算图，节省内存
           for param in model.parameters(): # param (tensor,grad)  在param将tensor减去grad
               param-=learning_rate*param.grad
        model.zero_grad()
    else:
        optimizer.step() # 使用优化方法更新参数
        optimizer.zero_grad() # 归零


