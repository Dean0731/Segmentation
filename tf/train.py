from tf.util.Config import *
from tf.util import TrainMethod
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model.fit  compile --->train
def main1():
    model.fit(data,validation_data=validation_data,epochs=num_epochs,callbacks=callback)
    tf.print("测试集开始测试".center(20,'*'))
    model.evaluate(test_data)
# model.train_on_batch compile    ----> train（自己可以控制）
def main2():
    TrainMethod.train_on_batch(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=num_epochs, log_dir=h5_dir)
    tf.print("测试集开始测试".center(20,'*'))
    TrainMethod.test_on_batch(model, test_data, len(test_data // 2))
# 自定义 compile 自定义train
def main3():
    TrainMethod.define_on_train(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=num_epochs, log_dir=h5_dir)
def main():
    tf.print("开始训练".center(20,'*'))
    main1()
    # model = main2()
    # model = main3()
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
if __name__ == '__main__':
    main()
