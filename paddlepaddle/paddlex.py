# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
# 图像预处理流程
target_size = 512
log = 'temp'
data_dir='/home/aistudio/data/data51222/'
train_list='/home/aistudio/data/data51222/data_train.txt'
val_list='/home/aistudio/data/data51222/data_val.txt'
test_list='/home/aistudio/data/data51222/data_test.txt'
label_list='/home/aistudio/data/data51222/data_label.txt'
from paddlex.seg import transforms

transforms = transforms.Compose([
    transforms.Resize(target_size=target_size,interp='NEAREST'),
    transforms.Normalize()
])

# 定义数据集
train_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=train_list,
    label_list=label_list,
    transforms=transforms,
    shuffle=True)
val_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=transforms)
test_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=test_list,
    label_list=label_list,
    transforms=transforms)
def main():
    # 训练模型
    model = pdx.seg.DeepLabv3p(num_classes=2)
    model = pdx.seg.FastSCNN(num_classes=2,use_bce_loss=True,use_dice_loss=True)
    model.train(
        num_epochs=40,
        train_dataset=train_dataset,
        train_batch_size=4,
        eval_dataset=val_dataset,
        learning_rate=0.01,
        save_interval_epochs=1,
        save_dir= log,
        use_vdl=True,
        # resume_checkpoint='',
    )
    model.evaluate(test_dataset)
def toPic():
    model = pdx.load_model(os.path.join(log,'best_model'))
    for x,y in test_dataset.file_list:
        result = model.predict(x)
        pdx.seg.visualize(x, result, weight=0.4, save_dir=log)
        break
if __name__ == '__main__':
    main()

    #!visualdl --logdir=output/vdl_log --port=8008,8040
