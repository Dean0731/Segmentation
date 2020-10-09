import paddle
class UNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.conv_1 = paddle.nn.Conv2d(3, 32,kernel_size=3,padding=0)
        self.bn = paddle.nn.BatchNorm2d(32)
        self.relu = paddle.nn.ReLU()


    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)

        return y
if __name__ == '__main__':
    device = paddle.set_device(paddle.device.get_device())
    paddle.disable_static(device)
    print("使用设备：",device)
    num_classes = 4
    model = paddle.Model(UNet(num_classes))
    model.summary((3, 160, 160))
