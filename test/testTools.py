import os
def compareFolder(file,file2):
    image = file
    label = file2
    image = os.listdir(image)
    label = os.listdir(label)
    print(len(label))
    print(image==label)

def compareFolder(file_less,file_much):
    """
    比较两个文件夹的差积 file_much-file_less
    """
    image = os.listdir(file_less)
    label = os.listdir(file_much)
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