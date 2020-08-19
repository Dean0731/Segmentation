import os
from util import Config,Tools,Dataset
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
@Tools.Decorator.timer(flag=True)
def main():
    target_size = (576,576)
    mask_size = (576,576)
    num_classes = 2
    batch_size = 2

    # 获取数据
    dataset = Dataset.Dataset(Config.Path.Shiyanshi_hu,target_size,mask_size,num_classes)
    tf.print("开始训练".center(20,'*'))
    model,callback,data,validation_data,test_data,train_step,val_step,test_step,num_classes,epochs,h5_dir = Config.getNetwork_Model("segnet", dataset, batch_size, target_size, num_classes)
    model = Config.complie(model, lr=0.001, num_classes=num_classes)
    model.fit(data, validation_data=validation_data,steps_per_epoch=train_step,validation_steps=val_step,epochs=epochs,callbacks=callback)
    model.save_weights(os.path.join(h5_dir, 'last.h5'))
    model.evaluate(test_data,steps=test_step)
    tf.print("训练结束".center(20,'*'))
if __name__ == '__main__':

    ret, time = main()
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    msg ="The job had cost about {}h{}m{}s".format(h,m,int(s))
    Tools.sendMessage(msg)
