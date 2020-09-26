import tf as tf
from tf.keras.layers import Input,Conv2D
from tf.keras import Model

input = Input((3072,3072,3))

x = Conv2D(filters=32,kernel_size=(11,11),strides=(3,3),padding="same")(input)
x = Conv2D(filters=32,kernel_size=(11,11),strides=(2,2),padding="same")(x)
x = Conv2D(filters=32,kernel_size=(11,11),strides=(2,2),padding="same")(x)
# x = Conv2D(filters=128,kernel_size=(11,11),strides=(2,2),padding="same")(x)


# x = Conv2D(filters=64,kernel_size=(11,11),padding="same")(x)
# x = Conv2D(filters=64,kernel_size=(11,11),padding="same")(x)
# x = Conv2D(filters=64,kernel_size=(11,11),strides=(2,2),padding="same")(x)
#
# x = Conv2D(filters=128,kernel_size=(11,11),padding="same")(x)
# x = Conv2D(filters=128,kernel_size=(11,11),padding="same")(x)
# x = Conv2D(filters=128,kernel_size=(11,11),strides=(2,2),padding="same")(x)

model = Model(inputs=input,outputs=x)
model.summary()
