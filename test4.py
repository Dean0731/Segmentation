from util.dataset import dataset_tools
dataset = dataset_tools.selectDataset('C',"{}_{}".format('tif',3072),parent='/home/dean/PythonWorkSpace/Segmentation/dataset')
img = dataset.getGernerator()
label = dataset.getGernerator(type='label_png')

first_batch = img.__next__()
print(len(first_batch))
print(first_batch[0].shape)

first_batch = label.__next__()
print(len(first_batch))
print(first_batch[0].shape)
#
for i in range(64):
    for j in range(64):
        print(first_batch[1,i,j,0],end='')
    print()