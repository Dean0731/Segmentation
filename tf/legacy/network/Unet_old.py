from tf import keras
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
up_conv_padding = "same"
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
    x = keras.layers.Conv2D(k,(kx,ky),strides=stride,padding=padding)(x)
    x = keras.layers.BatchNormalization(axis=chandim)(x)
    x = keras.layers.Activation('relu')(x)
    return x
@outer(Flag=log)
def upconv_module(x,k,kx,ky,stride,chandim=3,padding="same",x2=None):
    x = keras.layers.Conv2DTranspose(filters=k,kernel_size=(kx,ky),strides=stride,padding=padding)(x)
    x = keras.layers.concatenate([x,x2],axis=chandim)
    return x

def make_generator_model(width,height,channel):
    input = keras.layers.Input(shape=(width,height,channel))
    conv_1_1 = conv_module(input,filter,kernel_size,kernel_size,strides,padding=padding)
    conv_1_2 = conv_module(conv_1_1, filter, kernel_size, kernel_size, strides, padding=padding)
    maxPool_1 = keras.layers.MaxPool2D((pool_size,pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_1_2)

    conv_2_1 = conv_module(maxPool_1, filter*2, kernel_size, kernel_size, strides, padding=padding)
    conv_2_2 = conv_module(conv_2_1, filter*2, kernel_size, kernel_size, strides, padding=padding)
    maxPool_2 = keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_2_2)

    conv_3_1 = conv_module(maxPool_2, filter*4, kernel_size, kernel_size, strides, padding=padding)
    conv_3_2 = conv_module(conv_3_1, filter*4, kernel_size, kernel_size, strides, padding=padding)
    maxPool_3 = keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_3_2)

    conv_4_1 = conv_module(maxPool_3, filter*8, kernel_size, kernel_size, strides, padding=padding)
    conv_4_2 = conv_module(conv_4_1, filter*8, kernel_size, kernel_size, strides, padding=padding)
    #conv_4_2 = keras.layers.Dropout(0.5)(conv_4_2)
    maxPool_4 = keras.layers.MaxPool2D((pool_size, pool_size), strides=(pool_strides, pool_strides),padding=pool_padding)(conv_4_2)

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

    conv_10 = keras.layers.Conv2D(filters=class_num, kernel_size=1, strides=1, padding=padding,use_bias=use_bias)(conv_9_2)
    end = keras.layers.Activation("softmax")(conv_10)
    print(end)
    model = keras.models.Model(input, end)
    return model
if __name__ == '__main__':
    model = make_generator_model(576,576,3)
    model.summary()
    # model = make_generator_model(3072,3072,3)
    # model.summary()
