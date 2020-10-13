
import paddle
print(paddle.__version__)
paddle.disable_static()
#
# fc = paddle.to_tensor([0.8,0.9],dtype='float32')
# import math
# print(paddle.nn.Softmax()(fc))
# print(math.log(paddle.nn.Softmax()(fc).numpy()[0])/2)
# print(math.log(paddle.nn.Softmax()(fc).numpy()[1])/2)
#
# label = paddle.to_tensor([0],dtype='int64')
# import paddle.nn.functional as F
# out = F.softmax_with_cross_entropy(logits=fc, label=label)
# print(out)
#
import numpy as np
a = np.array([[
    [
        [0.6,0.6,0.6,0.6],
        [0.6,0.4,0.4,0.6],
        [0.6,0.4,0.4,0.6],
        [0.6,0.6,0.6,0.6],
    ],
    [
        [0.4,0.4,0.4,0.4],
        [0.4,0.6,0.6,0.4],
        [0.4,0.6,0.6,0.4],
        [0.4,0.4,0.4,0.4],
    ],
],
    [
        [
            [0.6,0.6,0.6,0.6],
            [0.6,0.4,0.4,0.6],
            [0.6,0.4,0.4,0.6],
            [0.6,0.6,0.6,0.6],
        ],
        [
            [0.4,0.4,0.4,0.4],
            [0.4,0.6,0.6,0.4],
            [0.4,0.6,0.6,0.4],
            [0.4,0.4,0.4,0.4],
        ],
    ]
])
b = np.array([[
    [
        [0,0,0,0],
        [0,0,1,1],
        [0,0,1,1],
        [0,0,0,0],
    ]
],
    [
        [
            [0,0,0,0],
            [0,0,1,1],
            [0,0,1,1],
            [0,0,0,0],
        ]
    ]
])

x = paddle.to_tensor(a)
y = paddle.to_tensor(b)
print(x.shape)
print(y.shape)
# x = paddle.flatten(x,2,-1)
# y = paddle.flatten(y,2,-1)
# x = paddle.transpose(x,perm=(0,2,1))
# y = paddle.transpose(y,perm=(0,2,1))
# print(x.shape)
# print(y.shape)
# m = paddle.metric.Accuracy()

import paddlepaddle.util.Config as C
m = C.MyAcc()

correct = m.compute(x, y)
m.update(correct)
res = m.accumulate()
print(res) # 0.75
