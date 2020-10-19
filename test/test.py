# Using 'none' reduction type.
import tensorflow as tf
label = tf.io.read_file(r'E:\DeepLearning\AI_dataset\dom\png_png\val\label_png\peizhuangcun51-dom\peizhuangcun51-dom_00001.png')
label = tf.image.decode_png(label,channels=1,dtype=tf.uint8)
label = tf.image.resize(label,(512,512),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
import util
util.printArray(label.numpy())

