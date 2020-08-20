import tensorflow as tf

data = tf.data.Dataset.range(10)
print(len(data))
# train = data.take(5)
# val = data.skip(5).take(3)
# test = data.skip(5+3)
#
# print(list(data.as_numpy_iterator()))
# print(list(train.as_numpy_iterator()))
# print(list(val.as_numpy_iterator()))
# print(list(test.as_numpy_iterator()))
s = tf.constant('abcdef')
print(dir(s))
