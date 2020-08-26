import keras
import numpy as np
import os
from PIL import Image
from unet.unet import mobilenet_unet
NCLASSES = 2
HEIGHT = 3072
WIDTH = 3072
BATCH_SIZE = 4
# 数据集txt
train_file = r'train.txt'
train_dir = r'/home/weiht/MDT/dzf/AI_dataset/DOM/大郭楼-DOM/image-3000/img'
label_dir = r'/home/weiht/MDT/dzf/AI_dataset/DOM/大郭楼-DOM/image-3000/label'
def generate_arrays_from_file(lines,batch_size):
    # 获取总的数据数量
    n = len(lines)
    i = 0
    while 1:
        X = []
        Y = []
        # 获取一个batch_size数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            img_name,lable_name = [name.strip() for name in lines[i].split(';')]
            # 读取图像
            img = Image.open(os.path.join(train_dir,img_name))
            img = img.resize((WIDTH,HEIGHT))
            img = np.array(img)
            X.append(img)

            img = Image.open(os.path.join(label_dir,lable_name))
            # img = img.resize((WIDTH,HEIGHT))
            img = img.resize((int(WIDTH/2),int(HEIGHT/2)))
            img = np.array(img)
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:] == c ).astype(int)
            #seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y.append(seg_labels)

def loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true,y_pred)
    return loss

# from utils import decorator
def main():
    # 权重信息，tensorboard文件保存路径
    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 获取model
    model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    # 打开数据集txt
    with open(train_file,'r') as f:
        lines = f.readlines()
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 设置验证集，训练集数量
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        # 保存路径
        os.path.join(log_dir,'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        monitor='val_loss',
        # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        save_weights_only=True,
        # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        save_best_only=True,
        # 每个checkpoint之间epochs间隔数量
        period= 1
    )
    # 早停
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    tensorboard = keras.callbacks.TensorBoard(log_dir),
    # 当被检测的量不再变化时，改变学习率
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        # lr = lr*factor
        factor=0.5,
        # 每当3个epoch 过去时，监测值不变触发学习率变化
        patience=3,
        verbose=1
    )
    model.compile(
        loss="sc",
        optimizer= keras.optimizers.Adam(lr=0.001),
        metrics=['acc']
    )
    model.fit(generate_arrays_from_file(lines[:num_train], BATCH_SIZE),
              steps_per_epoch=max(1, num_train//BATCH_SIZE),
              validation_data=generate_arrays_from_file(lines[num_train:], BATCH_SIZE),
              validation_steps=max(1, num_val//BATCH_SIZE),
              epochs=1,
              initial_epoch=0,
              callbacks=[checkpoint, reduce_lr,tensorboard])
    model.sample_weights(os.path.join(log_dir,'last.h5'))
if __name__ =="__main__":
    main()