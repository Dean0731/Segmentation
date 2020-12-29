# @Time     : 2020/8/25 17:40
# @File     : test_tf
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release

from tensorflow.python.client import device_lib
import tensorflow as tf
tf.get_logger().setLevel(3)

print("设备信息如下所示".center(20,'*'))
local_device = device_lib.list_local_devices()
[print(x) for x in local_device]
print("Tensorflow version:",tf.__version__)

# 2.4及之后
gpu = tf.config.list_physical_devices('GPU')
cpu = tf.config.list_physical_devices('GPU')
print("cpu:",cpu)
print("gpu:",gpu)

