# @Time     : 2020/8/25 17:24
# @File     : pytorch_for_nn_by_model
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import torch
import torch.nn as nn
class TwoLayerNet(nn.Module):
    def __init__(self,D_in,H,D_out): # 定义框架
        super(TwoLayerNet,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.linear2 = nn.Linear(H,D_out)
    def forward(self,x): # 前向传播过程
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred
N,D_in,H,D_out = 64,1000,100,10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

# model = nn.Sequential(
#     nn.Linear(D_in,H,bias=False),
#     nn.ReLU(),
#     nn.Linear(H,D_out,bias=False),
# )
model = TwoLayerNet(D_in,H,D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for t in range(500):
    # 前向传播
    y_pred = model(x)
    # 计算损失，需要损失函数
    loss = loss_fn(y_pred,y) # 均方误差函数，数学上是公式，在这里是计算图，这样才能求导数
    print(t,loss.item())
    # 反向传播,求梯度值
    loss.backward() # 将梯度值放在参数中

    optimizer.step() # 使用优化方法更新参数
    optimizer.zero_grad() # 归零


