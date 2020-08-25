import tf as tf
import numpy as np
from PIL import Image
import os
img_rows = 3072
img_cols = 3072
batchsize = 1
NCLASSES = 2
def getGenerator(dir):
    data_gen_args = dict(featurewise_center=False,
        samplewise_center=False, featurewise_std_normalization=False,
        samplewise_std_normalization=False, zca_whitening=False,
        zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
        height_shift_range=0.0, brightness_range=None, shear_range=0.0,
        zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest',
        cval=0.0, horizontal_flip=False, vertical_flip=False,
        rescale=None, preprocessing_function=None, data_format=None,
        validation_split=0.0)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    seed = 1
    image_generator = image_datagen.flow_from_directory(dir, target_size=(img_rows, img_cols),classes=['img'],
                                                        class_mode=None, seed=seed, batch_size=batchsize,
                                                        color_mode='rgb')
    mask_generator = mask_datagen.flow_from_directory(dir, target_size=(img_rows, img_cols),classes=['label'],
                                                      class_mode=None, seed=seed, batch_size=batchsize,
                                                      color_mode='grayscale')
    train_generator = zip(image_generator, mask_generator)
    return train_generator
def genGenerator2(dir):
    img_path = os.path.join(dir,'img')
    label_path = os.path.join(dir,'label')
    images_name = os.listdir(img_path)
    labels_name = os.listdir(label_path)
    # 获取总长度
    i=0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batchsize):
            # 从文件中读取图像
            img = Image.open(os.path.join(img_path,images_name[i]))
            img = img.resize((img_cols, img_rows))
            img = np.array(img)
            img = img / 255
            X_train.append(img)

            # 从文件中读取图像
            img = Image.open(os.path.join(label_path, labels_name[i]))
            img = img.resize((img_cols, img_rows))
            img = np.array(img)
            # seg_labels = np.zeros((img_cols, img_rows, NCLASSES))
            # for c in range(NCLASSES):
            #     seg_labels[:, :, c] = (img[:, :] == c).astype(int)
            # seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            seg_labels = np.reshape(img,(img_cols*img_rows,))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i=i+1
        yield (np.array(X_train), np.array(Y_train))
if __name__ == "__main__":
    dir = r'G:\AI_dataset\DOM\大郭楼-DOM\image-3000'
    generator = genGenerator2(dir)
    print(generator.__next__()[0].shape)
    print(generator.__next__()[1].shape)
    print(generator.__next__()[3].shape)
    print(generator.__next__()[4].shape)
    # print(x)
    # print(y)