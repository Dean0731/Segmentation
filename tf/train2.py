import os
import tf as tf
from util import Tools
from tf.util import Config1

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# model.fit  compile --->train
@Tools.Decorator.timer(flag=True)
def main():
    # 获取数据
    model = "mysegnet_3"
    target_size = (256,256)
    mask_size = target_size
    num_classes = 2

    model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir = Config1.getNetwork_Model(model, target_size, mask_size, num_classes)
    model = Config1.complie(model, lr=learning_rate, num_classes=num_classes)

    tf.print("开始训练".center(20,'*'))
    model.fit(data,validation_data=validation_data,epochs=epochs,callbacks=callback)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model.evaluate(test_data)
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
# model.train_on_batch compile    ----> train（自己可以控制）
@Tools.Decorator.timer(flag=True)
def main1():
    # 获取数据
    model = "mysegnet_3"
    target_size = (256,256)
    mask_size = target_size
    num_classes = 2

    model,learning_rate,callback,data,validation_data,test_data,epochs,h5_dir = Config1.getNetwork_Model(model, target_size, mask_size, num_classes)
    model = Config1.complie(model, lr=0.001, num_classes=num_classes)

    tf.print("开始训练".center(20,'*'))
    model = Config1.train_on_batch(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))
    tf.print("测试集开始测试".center(20,'*'))
    model = Config1.test_on_batch(model, test_data, len(test_data // 2))
    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
# 自定义 compile 自定义train
@Tools.Decorator.timer(flag=True)
def main2():
    # 获取数据
    model = "mysegnet_3"
    target_size = (256,256)
    mask_size = target_size
    num_classes = 2
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = Config1.getNetwork_Model(model, target_size, mask_size, num_classes)

    tf.print("开始训练".center(20,'*'))
    model = Config1.define_on_train(model, data, steps_per_epoch=len(data) // 2, validation_data=validation_data, validation_steps=len(validation_data // 2), epochs=epochs, log_dir=h5_dir)
    tf.print("训练结束".center(20,'*'))

    tf.print("保存模型".center(20,'*'))
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
if __name__ == '__main__':
    ret, seconds = main()
    msg ="The job had cost about {}h{}m{}s".format(*Tools.getSecondToTime(seconds))
    Tools.sendMessage(msg)