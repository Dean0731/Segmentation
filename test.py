from PIL import Image
import numpy as np
import util
def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z
path = r'C:\Users\root\OneDrive - bit.edu.cn\桌面\毕业设计\temp\test.jpg'
img = Image.open(path)
img = np.array(img)
img = np.expand_dims(img,axis=0)
print(img.shape)
img = img.transpose(0,3,1,2)
print(img.shape)
img = max_pooling_forward(img,(2,2))
print(img.shape)
img = img.transpose(0,2,3,1)
print(img.shape)
img = img[0,:,:,:]
seg_img = Image.fromarray(np.uint8(img)).convert('RGB')
seg_img.save(r"C:\Users\root\OneDrive - bit.edu.cn\桌面\毕业设计\temp\test-maxpool.jpg")