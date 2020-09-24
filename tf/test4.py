import os
import tensorflow as tf
from util import func
from tf.util import Config
from tf.util import TrainMethod
model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir, num_classes = Config.getNetwork_Model()

for x,y in test_data:
    y = y[0,:,:,1]
    for i in range(64):
        for j in range(64):
            print(int(y[i,j].numpy()),end='')
        print()
