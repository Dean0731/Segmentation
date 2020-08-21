import numpy as np
import tensorflow as tf
from util import Config, Dataset
from network import Model
np.set_printoptions(threshold = 1e6)
def main():
    target_size = (576,576)
    mask_size = (352,352)
    num_classes = 2
    batch_size = 2
    h5 = 'last.h5'
    dataset = Dataset.Dataset(Config.Path.lENOVO_PC,target_size,mask_size,num_classes)
    data,validation_data,test_data = dataset.getTrainValTestDataset()
    data = data.batch(batch_size)
    validation_data = validation_data.batch(batch_size)
    test_data =test_data.batch(batch_size)

    model = Model.getModel("segnet",target_size,n_labels=num_classes)
    print("model name:",model.name)
    model.load_weights(h5)
    model = Config.complie(model,lr=0.001,num_classes=num_classes)
    model.evaluate(test_data)

if __name__ == '__main__':
    main()