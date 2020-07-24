from PIL import Image
import numpy as np
import os
from network import Segnet,Unet,Mine_Segnet3
np.set_printoptions(threshold = 1e6)

def main():
    target_size = (576,576)
    num_classes = 2
    h5 = 'last.h5'
    set='trainSet'
    set='testSet'
    set='testSet'

    model = Segnet.Segnet(target_size[0], target_size[1], 3, n_labels=num_classes)
    model.load_weights(h5)

    dir = os.path.join('source\images\img',set)
    label_dir = os.path.join('source\images\img',"{}_pre_label".format(set))

    for img in os.listdir(dir):
        name = img
        img = Image.open(os.path.join(dir,img))
        img = img.resize(target_size)
        img = np.array(img)
        img = img.reshape(-1,target_size[0],target_size[1],3)
        pr = model.predict(img)[0]
        pr = pr.argmax(axis=2)
        pr[:,:] = (pr[:,:]==1) *255
        seg_img = Image.fromarray(np.uint8(pr)).convert('P')
        seg_img.save(os.path.join(label_dir,"{}-label.png".format(str(name).split('.')[0])))
        print(name)
if __name__ == '__main__':
    main()