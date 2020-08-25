import platform
import os
import numpy as np
from tf import keras
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
img_name, label_name = ('img', 'label_png')

def getGenerator(target_size=(64, 64), seed=7, batch_size=4, num_classes=2, parent_dir=None):
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
    image_generator = image_datagen.flow_from_directory(parent_dir, target_size=(576,576), classes=[img_name],
                                                        class_mode=None, seed=seed, batch_size=batch_size,
                                                        color_mode='rgb')
    label_generator = image_datagen.flow_from_directory(parent_dir, target_size=(576,576), classes=[label_name],
                                                        class_mode=None, seed=seed, batch_size=batch_size,
                                                        color_mode='grayscale')
    train_generator = zip(image_generator, label_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, num_classes)
        yield (img, mask)  # 用生成器进行迭代数据，可以传入model.fit_generator（）这个函数进行训练

def adjustData(img, mask, num_classes):
    #img = img / 255
    mask = mask[:, :, :, 0].astype(int) if (len(mask.shape) == 4) else mask[:, :, 0]
    new_mask = np.zeros(mask.shape + (num_classes,)).astype(int)  # (2, 416, 416, 2)
    for i, j in zip(range(num_classes), [0, 38]):
        new_mask[:, :, :, i] = (j == mask[:, :, :])
    mask = new_mask
    return (img, mask)

from util import timer
@timer(flag=True)
def main(model):
    tensofboard_dzf = keras.callbacks.TensorBoard(log_dir=event_dir)
    checkpoint_dzf = keras.callbacks.ModelCheckpoint(
        # 保存路径
        os.path.join(h5_dir,'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        monitor='val_loss',
        # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        save_weights_only=True,
        # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        save_best_only=True,
        # 每个checkpoint之间epochs间隔数量
        period=50
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy']
    )

    model.fit_generator(getGenerator((width, height), seed=7, batch_size=batch_size, parent_dir=train_dir),
                        steps_per_epoch=max(1, 208//batch_size),
                        validation_data=getGenerator((width, height), seed=7, batch_size=batch_size, parent_dir=val_dir),
                        validation_steps=max(1, 29//batch_size),
                        epochs=501,
                        callbacks=[checkpoint_dzf,tensofboard_dzf]
                        )
    model.save_weights(os.path.join(h5_dir, 'last.h5'))

if __name__ == '__main__':
    log_dir = os.path.join('source','logs')
    h5_dir = os.path.join(log_dir,'h5')
    event_dir = os.path.join(log_dir,'event')
    for i in [log_dir,h5_dir,event_dir]:
        if not os.path.exists(i):
            os.makedirs(i)
    batch_size = 4
    width = 576
    height = 576
    # from network.Unet import make_generator_model
    # model = make_generator_model(576, 576, 3)
    from tf.network.Segnet import segNetmodel
    main(segNetmodel)
