import util
from tf.util.Config import *
from tf.util.Config import model as mymodel
from tf.util import TrainMethod

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model.fit  compile --->train
def main1():
    tf.print("开始训练".center(20,'*'))
    model.fit(data,validation_data=validation_data,epochs=EPOCH_NUM,callbacks=callback)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model.evaluate(test_data)
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))

# model.train_on_batch compile    ----> train（自己可以控制）
def main2():
    tf.print("开始训练".center(20,'*'))
    model = TrainMethod.train_on_batch(mymodel, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model = TrainMethod.test_on_batch(model, test_data, len(test_data // 2))
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))


# 自定义 compile 自定义train
def main3():
    tf.print("开始训练".center(20,'*'))
    model = TrainMethod.define_on_train(mymodel, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))

    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))

@util.cls.Decorator.sendMessageWeChat()
@util.cls.Decorator.timer(flag=True)
def main():
    main1()
    # main2()
    # main3()
if __name__ == '__main__':
    main()
