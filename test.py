import  tensorflow as tf

acc = tf.keras.metrics.Accuracy()
acc.update_state([1,0,1,0,1,0],[1,1,1,1,1,1])
print(acc.result().numpy())