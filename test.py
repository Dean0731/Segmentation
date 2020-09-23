import tensorflow as tf
import math
a = math.log2(0.95)
b = math.log2(0.01)
c = math.log2(4)

print(a)
print(b)
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
print(loss.numpy())