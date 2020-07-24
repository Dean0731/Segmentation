# @Time     : 2020/7/21 15:23
# @File     : Model
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/7/21 Dean First Release
from network import Deeplabv3plus,Segnet,Unet,Xception
def getModel(name,shape,n_labels=2):
    name = name.lower()
    if name =='deeplabv3plus' :
        model = Deeplabv3plus.Deeplabv3(*shape,n_labels=n_labels)
    elif name == 'segnet':
        model = Segnet.Segnet(*shape,n_labels=n_labels)
    elif name == 'unet':
        model = Unet.Unet(*shape,n_labels=n_labels)
    elif name == 'xception':
        model = Xception.Xception(*shape,n_labels=n_labels)
    else:
        model = None
        print('未找到网络')
    return model
if __name__ == '__main__':
    shape=(576,576)
    channels = 3
    getModel('xception',(shape+(channels,)),n_labels=1000).summary()