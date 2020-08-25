from unet.unet import mobilenet_unet
from PIL import Image
import numpy as np
np.set_printoptions(threshold = 1e6)
import random
import copy
import os

random.seed(0)
NCLASSES = 2
HEIGHT = 3072
WIDTH = 3072


model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
#model.load_weights("logs/ep015-loss0.070-val_loss0.076.h5")

model.load_weights("./logs/last_4_20.h5")
img = Image.open("../images/image3.tif")
img = img.resize((WIDTH,HEIGHT))
old_img = copy.deepcopy(img)
orininal_w,orininal_h = img.size
img = np.array(img)
img = img/255
img = img.reshape(-1,HEIGHT,WIDTH,3)
print(img.shape)
pr = model.predict(img)[0]
#print(pr)
pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)
# for i in range(208):
#     for j in range(208):
#             print(pr[i,j],end='')
#     print()
pr[:,:] = (pr[:,:]==1) *255

seg_img = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h)).convert('P')

seg_img.save("../images/image3_last_4_20_3072.png")


