import paddle
from paddlepaddle.util import Callback
from paddle.static import InputSpec

inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
labels = [InputSpec([None, 1], 'int64', 'label')]

train_dataset = paddle.vision.datasets.MNIST(mode='train')

model = paddle.Model(paddle.vision.LeNet(classifier_activation=None),
                     inputs, labels)

optim = paddle.optimizer.Adam(0.001)
model.prepare(optimizer=optim,
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=(paddle.metric.Accuracy()))
callback = [Callback.Visual(log_dir="source/visual")]
model.fit(train_dataset, batch_size=64,epochs=2,callbacks=callback)



