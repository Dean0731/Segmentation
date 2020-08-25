# @Time     : 2020/8/25 15:12
# @File     : test
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
# import torch
# x = torch.randn(4,4)
# y = torch.randn(1)
# print(x)
# print(x.numpy())
# print(y.item())
# device = torch.device("cuda")
# y = torch.ones_like(x,device=device)
# x = x.to(device)
# z = x+y

# print(z)
# print(z.to("cpu",torch.double))

import torch
import torch.nn.functional as F
input = torch.tensor([
    [1,2,3],
    [4,5,6]
],dtype=torch.float32)
# each element in target has to have 0 <= value < C
target = torch.tensor([2,2])
pred = F.log_softmax(input)
print(pred)
print(pred.argmax(dim=1))
output = F.nll_loss(pred, target)
print(output)
