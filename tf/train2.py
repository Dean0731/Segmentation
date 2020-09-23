import os
import tensorflow as tf
from util import Tools
from tf.util import Config
from tf.util import TrainMethod

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model.fit  compile --->train
@Tools.Decorator.sendMessage()
@Tools.Decorator.timer(flag=True)
def main():
    model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir, num_classes = Config.getNetwork_Model()
    model = Config.complie(model, lr=learning_rate, num_classes=num_classes)

    tf.print("开始训练".center(20,'*'))
    model.fit(data,validation_data=validation_data,epochs=epochs,callbacks=callback)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model.evaluate(test_data)
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))


# model.train_on_batch compile    ----> train（自己可以控制）
@Tools.Decorator.sendMessage()
@Tools.Decorator.timer(flag=True)
def main1():
    model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir, num_classes = Config.getNetwork_Model()
    model = Config.complie(model, lr=0.001, num_classes=num_classes)

    tf.print("开始训练".center(20,'*'))
    model = TrainMethod.train_on_batch(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model = TrainMethod.test_on_batch(model, test_data, len(test_data // 2))
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))


# 自定义 compile 自定义train
@Tools.Decorator.sendMessage()
@Tools.Decorator.timer(flag=True)
def main2():
    model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir, num_classes = Config.getNetwork_Model()

    tf.print("开始训练".center(20,'*'))
    model = TrainMethod.define_on_train(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))

    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
if __name__ == '__main__':
    seconds = main()
