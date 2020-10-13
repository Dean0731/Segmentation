import paddle
print(paddle.__version__)
paddle.disable_static()
import paddle.nn.functional as F
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

# a = np.asarray([
#     [76,89],
#     [65,89],
#     [95,89],
# ])
# # b = np.asarray([
# #     [0],
# #     [0],
# #     [1],
# # ])
a = paddle.to_tensor(a,dtype="float32")
# b = paddle.to_tensor(b,dtype="float32")
x = paddle.nn.Softmax(axis=1)(a)
res = F.cross_entropy(a,b,reduction='mean')
print(res.numpy())

# for i in x:
#     print(i[0].numpy()+i[1].numpy())