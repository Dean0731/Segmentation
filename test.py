import numpy as np

a = np.array([
    [
        [
            [0.6, 0.6, 0.6, 0.6],
            [0.6, 0.4, 0.4, 0.6],
            [0.6, 0.4, 0.4, 0.6],
            [0.6, 0.6, 0.6, 0.6],
        ],
        [
            [0.4, 0.4, 0.4, 0.4],
            [0.4, 0.6, 0.6, 0.4],
            [0.4, 0.6, 0.6, 0.4],
            [0.4, 0.4, 0.4, 0.4],
        ],
    ],
    # [
    #     [
    #         [0.6, 0.6, 0.6, 0.6],
    #         [0.6, 0.4, 0.4, 0.6],
    #         [0.6, 0.4, 0.4, 0.6],
    #         [0.6, 0.6, 0.6, 0.6],
    #     ],
    #     [
    #         [0.4, 0.4, 0.4, 0.4],
    #         [0.4, 0.6, 0.6, 0.4],
    #         [0.4, 0.6, 0.6, 0.4],
    #         [0.4, 0.4, 0.4, 0.4],
    #     ],
    # ]
])
b = np.array([
    [
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ]
    ],
    # [
    #     [
    #         [0, 0, 0, 0],
    #         [0, 0, 1, 1],
    #         [0, 0, 1, 1],
    #         [0, 0, 0, 0],
    #     ]
    # ]
])

import tensorflow as tf
a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)
a = tf.transpose(a,(0,2,3,1))
b = tf.transpose(b,(0,2,3,1))
a = tf.keras.layers.Softmax()(a)
# b = tf.keras.layers.Softmax()(b)
print(a.shape)
print(b.shape)
print(tf.argmax(a,axis=-1))
y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# Using 'auto'/'sum_over_batch_size' reduction type.
scce = tf.keras.losses.SparseCategoricalCrossentropy()
print(scce(y_true, y_pred).numpy())
print(scce(b,a).numpy())


tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
