from PIL import Image
import numpy as np
np.set_printoptions(threshold = 1e6)
import os
NCLASSES = 2
HEIGHT = 2816
WIDTH = 2816
target_width = 352
target_height = 352

h5 = r'last.h5'
#h5 = 'ep064-loss0.060-val_loss0.293.h5'
set='trainSet'
set='testSet'
#set='testSet'
dir = os.path.join('source\images\img',set)
pre_dir = os.path.join('source\images\img',set)
label_dir = os.path.join('source\images\img',"{}_pre_label".format(set))

from network import Segnet,Unet,Segnet3
#model = Unet.Unet(576,576,3,n_labels=NCLASSES)
model = Segnet3.Segnet(HEIGHT,WIDTH,3,n_labels=NCLASSES)
model.load_weights(h5)
imgs = os.listdir(dir)
for img in imgs:
    name = img
    img = Image.open(os.path.join(dir,img))
    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    #img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]
    pr = pr.reshape((target_height,target_width,NCLASSES)).argmax(axis=2)
    pr[:,:] = (pr[:,:]==1) *255
    seg_img = Image.fromarray(np.uint8(pr)).convert('P')
    #seg_img = Image.fromarray(np.uint8(pr))
    seg_img.save(os.path.join(label_dir,"{}-label.png".format(str(name).split('.')[0])))
    print(name)
