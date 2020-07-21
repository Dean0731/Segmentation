import numpy as np
from tensorflow import keras
from util.dataset import Dataset
from util import Tools,Evaluate
from network import Segnet
np.set_printoptions(threshold = 1e6)
def main():
    target_size = (576,576)
    mask_size = (352,352)
    num_classes = 2
    batch_size = 2
    h5 = 'last.h5'
    dataset = 'M'
    print("dataset:",dataset)
    dataset = Dataset.selectDataset('M',"{}_{}".format('tif',target_size[0]))
    data,validation_data,test_data = dataset.getData(target_size=target_size,mask_size=mask_size,batch_size=batch_size)
    train_step,val_step,test_step = [dataset.getSize(type)//batch_size for type in ['train','val','testNetwork']]
    model = Segnet.Segnet(target_size[0],target_size[1],3,n_labels=num_classes)
    print("model name:",model.name)
    model.load_weights(h5)
    model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[Evaluate.MyMeanIOU(num_classes=2), "acc"]
                  #metrics=['acc']
    )
    model.evaluate(test_data,steps=test_step,)

if __name__ == '__main__':
    main()