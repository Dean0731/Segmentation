# @Time     : 2020/7/23 13:48
# @File     : test2
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/7/23 Dean First Release

import tensorflow as tf
import os
import json
import train
from util import Tools
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.16.6.209:20000", "172.16.7.240:20001"]
    },
    'task': {'type': 'worker', 'index': 1}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


@Tools.Decorator.timer(flag=True)
def main():
    with strategy.scope():
        model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = train.getNetwork_Model(False)
        model = train.complie(model,lr=0.001,num_classes=num_classes)
    model = train.fit(model,data,steps_per_epoch=train_step,validation_data=validation_data,validation_steps=val_step,epochs=epochs,callbacks=callback)
if __name__ == '__main__':
    main()

