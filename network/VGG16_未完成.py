# Pragram:
#
# History:
# 2020-08-07 Dean First Release
# Email:dean0731@qq.com

import tensorflow as tf


def VGG16(input,n_labels=2):
    x = input
    w1 = tf.random.normal([3,3,3,64])
    x = tf.nn.conv2d(input=x,filters=w1,strides=1,padding='SAME')
    w1 = tf.random.normal([3,3,64,64])
    x = tf.nn.conv2d(input=x,filters=w1,strides=1,padding='SAME')
    pool = tf.nn.max_pool2d(input=x,ksize=(2,2),strides=2,padding="SAME")
    x = pool
    w2 = tf.random.normal([3,3,64,128])
    x = tf.nn.conv2d(input=x,filters=w2,strides=1,padding='SAME')
    w2 = tf.random.normal([3,3,128,128])
    x = tf.nn.conv2d(input=x,filters=w2,strides=1,padding='SAME')
    pool2 = tf.nn.max_pool2d(input=x,ksize=(2,2),strides=2,padding="SAME")
    x = pool2
    w3 = tf.random.normal([3,3,128,256])
    x = tf.nn.conv2d(input=x,filters=w3,strides=1,padding='SAME')
    w3 = tf.random.normal([3,3,256,256])
    x = tf.nn.conv2d(input=x,filters=w3,strides=1,padding='SAME')
    w3 = tf.random.normal([1,1,256,256])
    x = tf.nn.conv2d(input=x,filters=w3,strides=1,padding='SAME')
    pool3 = tf.nn.max_pool2d(input=x,ksize=(2,2),strides=2,padding="SAME")
    x = pool3
    w4 = tf.random.normal([3,3,256,512])
    x = tf.nn.conv2d(input=x,filters=w4,strides=1,padding='SAME')
    w4 = tf.random.normal([3,3,512,512])
    x = tf.nn.conv2d(input=x,filters=w4,strides=1,padding='SAME')
    w4 = tf.random.normal([1,1,512,512])
    x = tf.nn.conv2d(input=x,filters=w4,strides=1,padding='SAME')
    pool4 = tf.nn.max_pool2d(input=x,ksize=(2,2),strides=2,padding="SAME")
    x = pool4
    w5 = tf.random.normal([3,3,512,512])
    x = tf.nn.conv2d(input=x,filters=w5,strides=1,padding='SAME')
    x = tf.nn.conv2d(input=x,filters=w5,strides=1,padding='SAME')
    x = tf.nn.conv2d(input=x,filters=w4,strides=1,padding='SAME')
    pool5 = tf.nn.max_pool2d(input=x,ksize=(2,2),strides=2,padding="SAME")
    x = pool5

    x = tf.reshape(x,shape=[-1,x.shape[1]*x.shape[2]*x.shape[3]])
    w1 = tf.Variable(tf.random.truncated_normal([x.shape[1], 4096], stddev=0.1))
    o1 = tf.matmul(x,w1)
    x = o1
    w2 = tf.Variable(tf.random.truncated_normal([x.shape[1], 4096], stddev=0.1))
    o2 = tf.matmul(x,w2)
    x = o2
    w3 = tf.Variable(tf.random.truncated_normal([x.shape[1], n_labels], stddev=0.1))
    o3 = tf.matmul(x,w3)
    x = o3
    softmax = tf.nn.softmax(x)

    return softmax

def metrics(y_true,y_pred):
    return tf.metrics.categorical_accuracy(y_true,y_pred)
op = tf.optimizers.Adam
if __name__ == '__main__':
    x = tf.random.normal([4,288,288,3])
    softmax = VGG16(x,1000)
    print(softmax.shape)
    for i in range(4):
        for j in range(1000):
            print(int(softmax.numpy()[i,j]),end='')
        print()
    for epoch in range(60):
        for i in range(20):
            x = None
            y_true = None
            with tf.GradientTape() as tape:
                y_pred = VGG16(n_labels=1000)
                loss = tf.losses.categorical_crossentropy(y_true=y_true,y_pred=y_pred)
            grads = tape.gradient(loss,)

