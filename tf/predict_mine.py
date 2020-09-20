from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tf.network import Model
np.set_printoptions(threshold = 1e6)
def getModel(h5,type,target_size,num_classes):
    if type=="save_weights":
        model = Model.getModel("segnet", target_size, n_labels=num_classes)
        model.load_weights(h5)
    elif type=="save":
        model = tf.keras.models.load_model(h5)
    return model
def main():
    target_size = (576,576)
    num_classes = 2
    h5,type = 'last.h5',"save_weight"
    set='trainSet'
    set='testSet'
    set='M-testSet'
    model = getModel(h5,type,target_size,num_classes)
    dir = os.path.join('../source/images/img', set)
    label_dir = os.path.join('../source/images/img', "{}_pre_label".format(set))
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
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
# 保存的是整个模型 save

if __name__ == '__main__':
    main()