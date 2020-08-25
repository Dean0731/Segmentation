import platform
import os
import numpy as np
import keras
import time
import numpy as np

from keras import backend as K
def Precision(y_true, y_pred):
    """精确率"""
    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp= K.sum(K.round(K.clip(y_pred, 0, 1))) # predicted positives
    precision = tp/ (pp+ K.epsilon())
    return precision

def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1))) # possible positives
    recall = tp / (pp + K.epsilon())
    return recall

def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

pla = platform.system()
if pla == 'Windows':
    train_dir = 'G:\AI_dataset\DOM\大郭楼-DOM\image-3000'
    val_dir = 'G:\AI_dataset\DOM\殷楼村56-dom\image-3000'
else:
    train_dir = r'/public1/data/weiht/dzf/AI_dataset/DOM/大郭楼-DOM/image-3000'
    val_dir = r'/public1/data/weiht/dzf/AI_dataset/DOM/殷楼村56-dom/image-3000'
if not os.path.exists(train_dir):
    print("{}不存在".format(train_dir))
    exit(-1)
img_name,label_name = ('img','label_png')
def getGenerator(target_size=(64,64),seed=7,batch_size=4,num_classes=2,parent_dir=None):
    data_gen_args = dict(featurewise_center=False,
                             samplewise_center=False, featurewise_std_normalization=False,
                             samplewise_std_normalization=False, zca_whitening=False,
                             zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                             height_shift_range=0.0, brightness_range=None, shear_range=0.0,
                             zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest',
                             cval=0.0, horizontal_flip=False, vertical_flip=False,
                             rescale=None, preprocessing_function=None, data_format=None,
                             validation_split=0.0)
    image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(parent_dir, target_size=target_size,classes=[img_name],
                                                        class_mode=None, seed=seed, batch_size=batch_size,
                                                        color_mode='rgb')
    label_generator = image_datagen.flow_from_directory(parent_dir, target_size=(576,576),classes=[label_name],
                                                        class_mode=None, seed=seed, batch_size=batch_size,
                                                        color_mode='grayscale')
    train_generator = zip(image_generator,label_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,num_classes,True)  # shape(size,3000,3000,3|1)
        yield (img,mask)    #用生成器进行迭代数据，可以传入model.fit_generator（）这个函数进行训练

def adjustData(img,mask,num_classes,flag_multi_class=True):
    """

    :param img:
    :param mask:
    :param flag_multi_class: 用来开启多分类
    :param num_class: 类别数量
    :return:
    """
    if (flag_multi_class):#如果多分类，在mask添加多层，每层对应一个类别
        img = img / 255
        mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_classes,)) .astype(int) # (2, 416, 416, 2)
        for i,j in zip(range(num_classes),[0,38]):
            new_mask[:,:,:,i]= (j==mask[:,:,:])
        mask = new_mask
    elif (np.max(img) > 1):#如果不是多分类，直接对img，mask进行操作，不难理解
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
def main():
    #from Unet_functional_same import model
    from mine_unet import make_generator_model
    model = make_generator_model(576,576,3)
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensofboard_dzf = keras.callbacks.TensorBoard(log_dir=log_dir)
    # checkpoint
    checkpoint_dzf = keras.callbacks.ModelCheckpoint(
        # 保存路径
        log_dir+'_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        monitor='val_loss',
        # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        save_weights_only=True,
        # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        save_best_only=True,
        # 每个checkpoint之间epochs间隔数量
        period= 100
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['acc',Recall,Precision,F1]
    )
    batch_size = 4
    model.fit_generator(getGenerator((576,576),seed=7,batch_size=batch_size,parent_dir=train_dir),
                        steps_per_epoch=max(1, 208//batch_size),
                        validation_data=getGenerator((576,576),seed=7,batch_size=batch_size,parent_dir=val_dir),
                        validation_steps=max(1, 29//batch_size),
                        epochs=1,
                        initial_epoch=0,
                        callbacks=[checkpoint_dzf])

    model.save_weights(os.path.join(log_dir,'last.h5'))
if __name__ == '__main__':
    main()



