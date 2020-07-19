import os
image = r'G:\AI_dataset\法国-房屋数据集2\AerialImage576\test\gt'
label = r'G:\AI_dataset\法国-房屋数据集2\AerialImage576\test\images'
image = os.listdir(image)
label = os.listdir(label)
print(len(label))
print(image==label)