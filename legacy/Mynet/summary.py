from unet.unet import mobilenet_unet
model = mobilenet_unet(2,3072,3072)
model.summary()