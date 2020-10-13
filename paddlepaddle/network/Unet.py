import paddle
class SeparableConv2d(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(SeparableConv2d, self).__init__()
        # 第一次卷积操作没有偏置参数
        self.conv_1 = paddle.nn.Conv2d(in_channels,
                                       in_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=in_channels,
                                       weight_attr=weight_attr,
                                       bias_attr=False,
                                       data_format=data_format)
        self.pointwise = paddle.nn.Conv2d(in_channels,
                                          out_channels,
                                          1,
                                          stride=1,
                                          padding=0,
                                          dilation=1,
                                          groups=1,
                                          weight_attr=weight_attr,
                                          data_format=data_format)

    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.pointwise(y)

        return y
class Encoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.relu = paddle.nn.ReLU()
        self.separable_conv_01 = SeparableConv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.bn = paddle.nn.BatchNorm2d(out_channels)
        self.separable_conv_02 = SeparableConv2d(out_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 padding='same')
        self.pool = paddle.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_conv = paddle.nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=2,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relu(inputs)
        y = self.separable_conv_01(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.separable_conv_02(y)
        y = self.bn(y)
        y = self.pool(y)

        residual = self.residual_conv(previous_block_activation)
        y = paddle.add(y, residual)

        return y
class Decoder(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.relu = paddle.nn.ReLU()
        self.conv_transpose_01 = paddle.nn.ConvTranspose2d(in_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding='same')
        self.conv_transpose_02 = paddle.nn.ConvTranspose2d(out_channels,
                                                           out_channels,
                                                           kernel_size=3,
                                                           padding='same')
        self.bn = paddle.nn.BatchNorm2d(out_channels)
        self.upsample = paddle.nn.Upsample(scale_factor=2.0)
        self.residual_conv = paddle.nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relu(inputs)
        y = self.conv_transpose_01(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.conv_transpose_02(y)
        y = self.bn(y)
        y = self.upsample(y)

        residual = self.upsample(previous_block_activation)
        residual = self.residual_conv(residual)

        y = paddle.add(y, residual)

        return y
class UNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.conv_1 = paddle.nn.Conv2d(3, 32,
                                       kernel_size=3,
                                       stride=2,
                                       padding='same')
        self.bn = paddle.nn.BatchNorm2d(32)
        self.relu = paddle.nn.ReLU()

        in_channels = 32
        self.encoders = []
        self.encoder_list = [64, 128, 256]
        self.decoder_list = [256, 128, 64, 32]

        # 根据下采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.encoder_list:
            block = self.add_sublayer('encoder_%s'.format(out_channels),
                                      Encoder(in_channels, out_channels))
            self.encoders.append(block)
            in_channels = out_channels

        self.decoders = []

        # 根据上采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.decoder_list:
            block = self.add_sublayer('decoder_%s'.format(out_channels),
                                      Decoder(in_channels, out_channels))
            self.decoders.append(block)
            in_channels = out_channels

        self.output_conv = paddle.nn.Conv2d(in_channels,
                                            num_classes,
                                            kernel_size=3,
                                            padding='same')
        self.softmax = paddle.nn.Softmax()
    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)

        for encoder in self.encoders:
            y = encoder(y)

        for decoder in self.decoders:
            y = decoder(y)

        y = self.output_conv(y)

        return self.softmax(y)
if __name__ == '__main__':
    from paddle.static import InputSpec

    paddle.disable_static()
    num_classes = 2
    model = paddle.Model(UNet(num_classes=num_classes))
    model.summary((3, 512, 512))
