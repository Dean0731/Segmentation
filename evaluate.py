import numpy as np
from tensorflow import keras
np.set_printoptions(threshold = 1e6)
from util.dataset import CountrySide

num_classes = 2
batch_size = 16
target_width = 576
target_height = 576
mask_width = 576
mask_height = 576
val_step = 6
test_step = val_step+1

h5 = '/public1/data/weiht/dzf/workspace/Segmentation/source/logs-2020-5-25 16:36:24/h5/ep100-loss0.040-val_loss0.301.h5'
from network import Segnet

model = Segnet.Segnet(576,576,3,n_labels=num_classes)
model.load_weights(h5)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=[CountrySide.MyMeanIOU(num_classes=2), "acc"]
    #metrics=['acc']
)
model.evaluate(CountrySide.getGenerator(target_size=(target_width, target_height), mask_size=(mask_width, mask_height), batch_size=batch_size, data_type='test', flag=False),
               verbose=2,
               steps=test_step,
               )