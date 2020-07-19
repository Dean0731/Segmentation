# Pragram:
# 	查看网络大小，最后网络大小表示batch_size为1，
# History:
# 2020-07-04 Dean First Release
# Email:dean0731@qq.com

# from keras import applications
# model = applications.VGG16(input_shape=(576,576,3),include_top=False,weights=None)
# print("无全连接层总参数量:",model.count_params())
# model = applications.VGG16(input_shape=(576,576,3),include_top=True,weights=None)
# print("有全连接层总参数量:",model.count_params())


from network import Unet,Segnet,Segnet2,Segnet3,Segnet4
def networkSize(model):
    model = model(width,height,channel,n_labels=2)
    print("总参数量:",model.count_params())
    all_params_memory = 0
    all_feature_memory = 0
    for num,layer in enumerate(model.layers):
        #训练权重w占用的内存
        params_memory = layer.count_params()*4/(1024*1024)
        all_params_memory = all_params_memory + params_memory
        #特征图占用内存
        feature_shape = layer.output_shape
        feature_size=1
        for i in range(1,len(feature_shape)):
            feature_size = feature_size*feature_shape[i]
        feature_memory = feature_size*4/(1024*1024)
        print("layer:{}".format(num).ljust(10,' '),
              "特征图占用内存:{}".format(feature_shape).ljust(33,' '),"{}".format(feature_size).ljust(10,' '),"{}M".format(str(feature_memory)).ljust(10,' '),
              "训练权重w占用的内存:{}".format(layer.name).ljust(33,' '),"{}".format(layer.count_params()).ljust(10,' '),"{}M".format(str(params_memory))
              )
        all_feature_memory = all_feature_memory + feature_memory
    print()
    print("网络权重W占用总内存:",str(all_params_memory)+"M","网络特征图占用总内存:",str(all_feature_memory)+"M")
    print("网络总消耗内存:",str(all_params_memory+all_feature_memory)+"M")
def segnetSize(model):
    model = model(width,height,channel,n_labels=2)
    print("总参数量:",model.count_params())
    all_params_memory = 0
    all_feature_memory = 0
    for num,layer in enumerate(model.layers):
        #训练权重w占用的内存
        params_memory = layer.count_params()*4/(1024*1024)
        all_params_memory = all_params_memory + params_memory
        #特征图占用内存
        feature_shape = layer.output_shape
        if type(feature_shape) is list :
            feature_shape = feature_shape[0]
        feature_size=1
        for i in range(1,len(feature_shape)):
            feature_size = feature_size*feature_shape[i]
        feature_memory = feature_size*4/(1024*1024)
        print("layer:{}".format(num).ljust(10,' '),
              "特征图占用内存:{}".format(feature_shape).ljust(33,' '),"{}".format(feature_size).ljust(10,' '),"{}M".format(str(feature_memory)).ljust(10,' '),
              "训练权重w占用的内存:{}".format(layer.name).ljust(33,' '),"{}".format(layer.count_params()).ljust(10,' '),"{}M".format(str(params_memory))
              )
        all_feature_memory = all_feature_memory + feature_memory
    print()
    print("网络权重W占用总内存:",str(all_params_memory)+"M","网络特征图占用总内存:",str(all_feature_memory)+"M")
    print("网络总消耗内存:",str(all_params_memory+all_feature_memory)+"M")
if __name__ == '__main__':
    pass
    (width,height,channel)=(2816,2816,3)
    #(width,height,channel)=(576,576,3)
    #segnetSize(Segnet.Segnet)
    #segnetSize(Segnet2.Segnet)
    segnetSize(Segnet4.Segnet)