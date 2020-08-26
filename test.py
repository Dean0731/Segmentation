# @Time     : 2020/8/26 14:34
# @File     : test
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from torch import  nn
import torch
import  numpy as np
y = torch.tensor([
    [
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ],
    [
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ]
])
y_pred = torch.tensor([
    [
        [0,0,0,0],
        [0,0,1,1],
        [0,0,1,1],
        [0,0,0,0],
    ],
    [
        [0,0,0,0],
        [0,0,1,0],
        [0,0,1,0],
        [0,0,0,0],
    ]
])
# iou = 0
# for i,j in zip(y,y_pred):
#     a = j.sum()
#     b = i.sum()
#     j[j==0]=2
#     c = i.eq(j).sum()
#     t = c.item()/(a+b-c).item()
#     iou = t+iou
#     print(t)
#
# print(iou/2)

a = torch.tensor(5)
b = torch.tensor(2)
print(a/b)