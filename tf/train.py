import util
import os

import tensorflow as tf
from tf.util.Config import config
from tf.util import TrainMethod

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model.fit  compile --->train
def main1():
    config.model.fit(config.data,validation_data=config.validation_data,epochs=config.EPOCH_NUM,callbacks=config.callback)

    tf.print("测试集开始测试".center(20,'*'))
    config.model.evaluate(config.test_data)
    return config.model

# model.train_on_batch compile    ----> train（自己可以控制）
def main2():
    model = TrainMethod.train_on_batch(config.model, config.data, steps_per_epoch=len(config.data) // 2, validation_data=config.validation_data, validation_steps=len(config.validation_data // 2), epochs=config.epochs, log_dir=config.h5_dir)
    tf.print("测试集开始测试".center(20,'*'))
    model = TrainMethod.test_on_batch(model, config.test_data, len(config.test_data // 2))
    return model

# 自定义 compile 自定义train
def main3():
    model = TrainMethod.define_on_train(config.model, config.data, steps_per_epoch=len(config.data) // 2, validation_data=config.validation_data, validation_steps=len(config.validation_data // 2), epochs=config.epochs, log_dir=config.h5_dir)
    return model


@util.cls.Decorator.sendMessageWeChat()
@util.cls.Decorator.timer(flag=True)
def main():
    tf.print("开始训练".center(20,'*'))
    model = main1()
    # model = main2()
    # model = main3()
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(config.h5_dir, 'last.h5'))
if __name__ == '__main__':
    main()
