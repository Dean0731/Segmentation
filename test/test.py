import tf as tf
y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
print(loss.numpy())