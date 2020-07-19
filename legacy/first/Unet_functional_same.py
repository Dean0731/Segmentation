import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from legacy.first.generator import *
# 卷积核数量
filter=64
# 卷积核大小
kernel_size=3
# 卷积步长
strides=1
# 方式
padding="same"
# 偏执
use_bias=False
# 激活函数
activation="relu"
# 池化大小
pool_size = 2
pool_strides = 2
pool_padding = "same"
class_num =2
up_conv_size =2
up_conv_strides = 2
up_conv_padding = "valid"
log = False
def outer(Flag):
    def panduan(f):
        def inner(*args,**kwargs):
            conv = f(*args,**kwargs)
            if Flag:
                print(conv)
            return conv
        return inner
    return panduan
@outer(Flag=log)
def conv_module(x,k,kx,ky,stride,chandim=-1,padding="same"):
    #conv-bn-relu
    x = tf.keras.layers.Conv2D(k,(kx,ky),strides=stride,padding=padding)(x)
    x = tf.keras.layers.BatchNormalization(axis=chandim)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
@outer(Flag=log)
def upconv_module(x,k,kx,ky,stride,chandim=3,padding="same",x2=None):
    x = tf.keras.layers.Conv2DTranspose(filters=k,kernel_size=(kx,ky),strides=stride,padding=padding)(x)
    x = tf.keras.layers.concatenate([x,x2],axis=chandim)
    return x

def make_generator_model(width,height,channel):
    input = tf.keras.layers.Input(shape=(width,height,channel))
    conv_1_1 = conv_module(input,filter,kernel_size,kernel_size,strides,padding=padding)
    conv_1_2 = conv_module(conv_1_1, filter, kernel_size, kernel_size, strides, padding=padding)
    maxPool_1 = tf.keras.layers.MaxPool2D((pool_size,pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_1_2)

    conv_2_1 = conv_module(maxPool_1, filter*2, kernel_size, kernel_size, strides, padding=padding)
    conv_2_2 = conv_module(conv_2_1, filter*2, kernel_size, kernel_size, strides, padding=padding)
    maxPool_2 = tf.keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_2_2)

    conv_3_1 = conv_module(maxPool_2, filter*4, kernel_size, kernel_size, strides, padding=padding)
    conv_3_2 = conv_module(conv_3_1, filter*4, kernel_size, kernel_size, strides, padding=padding)
    maxPool_3 = tf.keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_3_2)

    conv_4_1 = conv_module(maxPool_3, filter*8, kernel_size, kernel_size, strides, padding=padding)
    conv_4_2 = conv_module(conv_4_1, filter*8, kernel_size, kernel_size, strides, padding=padding)
    #conv_4_2 = tf.keras.layers.Dropout(0.5)(conv_4_2)
    maxPool_4 = tf.keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_4_2)

    conv_5_1 = conv_module(maxPool_4, filter*16, kernel_size, kernel_size, strides, padding=padding)
    conv_5_2 = conv_module(conv_5_1, filter*16, kernel_size, kernel_size, strides, padding=padding)

    upconv_1 = upconv_module(conv_5_2,filter*8,up_conv_size,up_conv_size,up_conv_strides,padding=up_conv_padding,x2=conv_4_2)
    conv_6_1 = conv_module(upconv_1, filter * 8, kernel_size, kernel_size, strides, padding=padding)
    conv_6_2 = conv_module(conv_6_1, filter * 8, kernel_size, kernel_size, strides, padding=padding)

    upconv_2 = upconv_module(conv_6_2, filter * 4,up_conv_size,up_conv_size, up_conv_strides, padding=up_conv_padding, x2=conv_3_2)
    conv_7_1 = conv_module(upconv_2, filter * 4, kernel_size, kernel_size, strides, padding=padding)
    conv_7_2 = conv_module(conv_7_1, filter * 4, kernel_size, kernel_size, strides, padding=padding)

    upconv_3 = upconv_module(conv_7_2, filter * 2, up_conv_size,up_conv_size, up_conv_strides, padding=up_conv_padding,x2=conv_2_2)
    conv_8_1 = conv_module(upconv_3, filter * 2, kernel_size, kernel_size, strides, padding=padding)
    conv_8_2 = conv_module(conv_8_1, filter * 2, kernel_size, kernel_size, strides, padding=padding)

    upconv_4 = upconv_module(conv_8_2, filter , up_conv_size,up_conv_size, up_conv_strides, padding=up_conv_padding, x2=conv_1_2)
    conv_9_1 = conv_module(upconv_4, filter , kernel_size, kernel_size, strides, padding=padding)
    conv_9_2 = conv_module(conv_9_1, filter , kernel_size, kernel_size, strides, padding=padding)

    conv_10 = tf.keras.layers.Conv2D(filters=class_num, kernel_size=1, strides=1, padding=padding,use_bias=use_bias)(conv_9_2)
    reshape = tf.keras.layers.Reshape((width*height,2))(conv_10)
    end = tf.keras.layers.Softmax()(reshape)

    model = tf.keras.models.Model(input, end)
    return model
dir = r'G:\AI_dataset\DOM\大郭楼-DOM\image-3000'
#image_generator = getGenerator(dir)
image_generator = genGenerator2(dir)
model = make_generator_model(3072,3072,3)
model.summary()
# logdir = r'D:\desktop\Files\Workspace\PythonWorkSpace\DeepLearning\Segmentation\event'
# callbacks = [
#         tf.keras.callbacks.TensorBoard(logdir),
#         #tf.keras.callbacks.ModelCheckpoint(output_model_file,save_best_only = True),  # 默认保存最近一次训练,True表示保存效果最好的
#         #tf.keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-5)  # 提前结束 当阈值低于1e-3时记录一次,5次后停止
#     ]
def loss(y_true, y_pred):
    crossloss = tf.keras.backend.binary_crossentropy(y_true,y_pred)
    loss = 4 * tf.keras.backend.sum(crossloss)/3072/3072
    return loss
model.compile(loss="categorical_crossentropy",
        optimizer = "adam",  # 若loss太低,可能是算法的问题,换用优化过的梯度下降算法
        metrics = ['accuracy'])
model.fit(image_generator, steps_per_epoch=50, epochs=1, verbose=1)
