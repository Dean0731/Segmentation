import os
from util import Config,Tools
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@Tools.Decorator.timer(flag=True)
def main():
    model = "segnet"
    target_size = (576,576)
    mask_size = (576,576)
    num_classes = 2

    model,learning_rate,callback,train_data,validation_data,test_data,epochs,h5_dir = Config.getNetwork_Model(model,target_size,mask_size,num_classes)
    model = Config.complie(model, lr=learning_rate, num_classes=num_classes)

    tf.print("开始训练".center(20,'*'))
    model.fit(train_data,validation_data=validation_data,epochs=epochs,callbacks=callback)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model.evaluate(test_data)
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))


if __name__ == '__main__':
    ret, seconds = main()
    msg ="The job had cost about {}h{}m{}s".format(*Tools.getSecondToTime(seconds))
    Tools.sendMessage(msg)