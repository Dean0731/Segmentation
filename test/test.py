import os
image = r'G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset2\Training Set\Target maps'
label = r'G:\AI_dataset\马萨诸塞州-房屋数据集1\Massachusetts Buildings Dataset2\Training Set\Input images'
image = os.listdir(image)
label = os.listdir(label)
print(len(image))
print(len(label))
i,j=0,0
with open('temp.txt','w') as f:
    for _ in label:
        if label[i]==image[j]:
            i = i+1
            j = j+1
        else:
            f.write(label[i]+"\n")
            i = i+1
