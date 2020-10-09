# @Time     : 2020/7/21 17:31
# @File     : test2
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/7/21 Dean First Release
# def Precision(y_true, y_pred):
#     y_true,y_pred = tf.argmin(y_true,axis=-1),tf.argmin(y_pred,axis=-1)
#     sk.metrics.precision_score(y_true, y_pred)
#     TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # TP
#     N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0))) # N
#     TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
#     FP=N-TN
#     precision = TP / (TP + FP + tf.cast(K.epsilon(),tf.int64)) # TP/P
#     return precision
#
# def Recall(y_true, y_pred):
#     y_true,y_pred = tf.argmin(y_true,axis=-1),tf.argmin(y_pred,axis=-1)
#     TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # TP
#     P=K.sum(K.round(K.clip(y_true, 0, 1)))
#     FN = P-TP # FN=P-TP
#     recall = TP / (TP + FN + tf.cast(K.epsilon(),tf.int64)) # TP/(TP+FN)
#     return recall
from tf import train

target_size = (576,576)
mask_size = (576,576)
num_classes = 2
batch_size = 4
dataset = train.selectDataset('A', "{}_{}".format('tif', 576), parent='E:\Workspace\PythonWorkSpace\Segmentation\dataset')
data,validation_data,test_data = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)
data1 = data.__next__()
data1 = test_data.__next__()
data1 = validation_data.__next__()
# img,label = data1[0],data1[1]
# # print(img.shape,label.shape)
# img = img[0,:,:,:]
# label = label[1,:,:,:]
# print(label.shape)
# import numpy
# # label = numpy.argmax(label,axis=2)
# for i in range(576):
#     for j in range(576):
#         print(label[i,j,1],end='')
#     print()
# print(label.shape)