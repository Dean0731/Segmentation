import paddle
import numpy as np

paddle.disable_static()  # Now we are in imperative mode

all = paddle.to_tensor([])
a = paddle.to_tensor([1,])
b = paddle.to_tensor([1,])
c = paddle.tensor.stack(x=[a,b])
print(c.shape)