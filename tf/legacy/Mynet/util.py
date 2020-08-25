from PIL import Image
import numpy as np
def img():
    img = Image.open(r"D:\desktop\Files\Workspace\PythonWorkSpace\DeepLearning\U-net\images\work_json\label.png")
    print(img.mode)
    img = np.array(img)
    print(img.shape)
    seg_labels = np.zeros((599,930,2))
    for c in range(2):
        seg_labels[: , : , c ] = (img[:,:] == c ).astype(int)
    # for i in range(599):
    #     for j in range(930):
    #         print(int(seg_labels[i][j][1]),end="")
    #     print()
    print(seg_labels.shape)
def myrequest(url):
    import requests
    ret = requests.get(url)
    return ret.content.decode("utf8")
print(myrequest('https://pan.dean0731.top/myplugins/edit/index.php'))