import numpy as np
import torch,os
from PIL import Image

from pytorch.network.deeplabv3plus import modeling
from pytorch.util.Dataset import Dataset
from pytorch.util import Transform

def getModel(device,pt):
    # model = Segnet.Segnet2(3,2)
    # model.load_state_dict(torch.load(r'./source/mnist_cnn.pt',map_location=device))
    model = modeling.deeplabv3plus_resnet50(2, pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(pt,map_location=device))
    return model
def main():
    color = 2
    batch_size = 4
    device = torch.device("cpu")
    weight = r'C:\Users\root\OneDrive - bit.edu.cn\桌面\毕业设计\source\last.pt'
    model = getModel(device,weight)
    target_size = (512,512)
    dir = r'C:\Users\root\OneDrive - bit.edu.cn\桌面\毕业设计\source\images'
    test_dataset = Dataset(r'C:\Users\root\OneDrive - bit.edu.cn\桌面\毕业设计\source\data.txt',transform=Transform.getTransforms(target_size),target_transform= Transform.getTargetTransforms(target_size))
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_dataloader,start=0):
            data,target = data.to(device),target.to(device)
            pred = model(data) # batch_size * 10
            if hasattr(pred,"get"):
                if pred.get('out') is not None:
                    pred = pred.get('out')
            pred = torch.argmax(pred,dim=1)
            for id,i in enumerate(pred,start=0):
                image = Image.open(test_dataset.imgs[batch_size*idx+id][0]).resize(target_size)
                image = np.array(image)
                image[:,:,color] = image[:,:,color] - np.multiply(image[:,:,color],np.expand_dims(i.numpy(),2)[:,:,0])

                i[:,:] = (i[:,:]==1) *255
                seg_img = Image.fromarray(np.uint8(i)).convert('P')
                seg_img.save(os.path.join(dir,"{}-{}-{}.png").format(model._get_name(),idx,id))


                seg_img = Image.fromarray(np.uint8(image)).convert('RGB')
                seg_img.save(os.path.join(dir,"{}-RGB-{}-{}.png").format(model._get_name(),idx,id))
                print(idx,id)
if __name__ == '__main__':
    main()
