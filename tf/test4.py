import os
import tensorflow as tf
from util import Tools
from tf.util import Config
from tf.util import TrainMethod
model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir, num_classes = Config.getNetwork_Model()
model = Config.complie(model, lr=0.001, num_classes=num_classes)

tf.print("开始训练".center(20,'*'))

epochs= len(data) // 2,
validation_data=validation_data
validation_steps=len(validation_data // 2)
epochs=epochs
log_dir=h5_dir
