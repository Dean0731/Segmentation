import paddle

print(paddle.__version__)
paddle.disable_static()
train_dataset = paddle.vision.datasets.MNIST(mode='train', chw_format=False)
eval_dataset =  paddle.vision.datasets.MNIST(mode='test', chw_format=False)
print(train_dataset.__len__())
print(eval_dataset.__len__())
mnist = paddle.nn.Sequential(
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
# 开启动态图模式
paddle.disable_static()

# 预计模型结构生成模型实例，便于进行后续的配置、训练和验证
model = paddle.Model(mnist)

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(parameters=mnist.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
from paddlepaddle.util.Callback import Visual
# 开始模型训练
calllbacks = [Visual(),paddle.callbacks.ProgBarLogger(100)]
model.fit(train_dataset,
          epochs=10,
          eval_data= eval_dataset,
          batch_size=32,
          verbose=2,callbacks=calllbacks)
print("开始test")
# model.evaluate(val_dataset,log_freq=100,callbacks=calllbacks)