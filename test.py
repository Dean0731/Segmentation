# @Time     : 2020/8/26 17:18
# @File     : Segnet
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from torch import nn
import torch

pool_args = {"kernel_size":(2,2),"stride":(2,2),"return_indices":True}
unpool_args = {"kernel_size":(2,2),"stride":(2,2)}
activate = 'relu'
kernel_size = 3
layers = [
    ("conv",(64,kernel_size)),("bn",64),("activate",activate),("conv",(64,kernel_size)),("bn",64),("activate",activate),("pool",pool_args),
    ("conv",(128,kernel_size)),("bn",128),("activate",activate),("conv",(128,kernel_size)),("bn",128),("activate",activate),("pool",pool_args),
    ("conv",(256,kernel_size)),("bn",256),("activate",activate),("conv",(256,kernel_size)),("bn",256),("activate",activate),("conv",(256,1)),("bn",256),("activate",activate),("pool",pool_args),
    ("conv",(512,kernel_size)),("bn",512),("activate",activate),("conv",(512,kernel_size)),("bn",512),("activate",activate),("conv",(512,1)),("bn",512),("activate",activate),("pool",pool_args),
    ("conv",(512,kernel_size)),("bn",512),("activate",activate),("conv",(512,kernel_size)),("bn",512),("activate",activate),("conv",(512,1)),("bn",512),("activate",activate),("pool",pool_args),
    ("unpool",unpool_args), ("conv",(512,kernel_size)),("bn",512),("activate",activate), ("conv",(512,kernel_size)),("bn",512),("activate",activate), ("conv",(512,kernel_size)),("bn",512),("activate",activate),
    ("unpool",unpool_args), ("conv",(512,kernel_size)),("bn",512),("activate",activate), ("conv",(512,kernel_size)),("bn",512),("activate",activate), ("conv",(256,kernel_size)),("bn",256),("activate",activate),
    ("unpool",unpool_args), ("conv",(256,kernel_size)),("bn",256),("activate",activate), ("conv",(256,kernel_size)),("bn",256),("activate",activate), ("conv",(128,kernel_size)),("bn",128),("activate",activate),
    ("unpool",unpool_args), ("conv",(128,kernel_size)),("bn",128),("activate",activate),("conv",(64,kernel_size)),("bn",64),("activate",activate),
    ("unpool",unpool_args), ("conv",(64,kernel_size)),("bn",64),("activate",activate), ("conv",(2,1)),("bn",2),("activate",activate),
]
class MaxPoolIndex(nn.Module):
    def __init__(self, kernel_size,stride,return_indices):
        super(MaxPoolIndex, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.return_indices = return_indices
    def forward(self, x):
        return nn.functional.max_pool2d(x,self.kernel_size,self.stride,self.return_indices)

class MaxUnPoolIndex(nn.Module):
    def __init__(self, kernel_size,stride):
        super(MaxUnPoolIndex, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        return nn.functional.max_unpool2d(*x,self.kernel_size,self.stride)

class Segnet2(nn.Module):
    def __init__(self,input_channel,num_label,**kwargs):
        super(Segnet2,self).__init__(**kwargs)
    def forword(self,x):
        return x-x.mean()


if __name__ == '__main__':
    model = Segnet2(3,2)
    from torchsummary import summary
    summary(model,(3,512,512))