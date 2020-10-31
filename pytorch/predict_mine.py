from PIL import Image
import numpy as np
import os
import torch
from pytorch.network import Segnet,Deeplabv3
def getModel(h5,type):
    device = torch.device("cpu")
    model = torch.load(h5,map_location=device)
    if type =="save_weights":
        # model = Segnet.Segnet(3,2).load_state_dict(model).to(torch.device("cpu"))
        model = Deeplabv3.deeplabv3_resnet50(num_classes=2).load_state_dict(model)
    return model
Deeplabv3.deeplabv3_resnet50(num_classes=2).lo
def main():
    h5,type = r'../source/last.pt',"save_weight"
    model = getModel(h5,type)
    target_size = (512,512)
    dir = r'../source/images'

    for id,img in enumerate(os.listdir(dir)):
        img = Image.open(os.path.join(dir,img))
        img = img.resize(target_size)
        img = np.array(img)
        pr = model(img)
        # pr = pr.argmax(axis=2)
        # pr[:,:] = (pr[:,:]==1) *255
        # seg_img = Image.fromarray(np.uint8(pr)).convert('P')
        # seg_img.save(os.path.join("{}-label.png".format(id)))
# 保存的是整个模型 save

if __name__ == '__main__':
    main()