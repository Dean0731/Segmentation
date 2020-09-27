# @Time     : 2020/9/27 16:54
# @File     : train
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/9/27 Dean First Release

import os
from paddlepaddle.util.Config import *
def main():
    # 训练的轮数
    print('开始训练...')
    #两种方法，用两个不同的路径分别保存训练的模型
    #model_save_dir = "/home/aistudio/data/model_vgg"
    model_save_dir = "/home/aistudio/data/model_cnn"
    for pass_id in range(EPOCH_NUM):
        train_cost = 0
        for batch_id, data in enumerate(train_reader()):                         #遍历train_reader的迭代器，并为数据加上索引batch_id
            train_cost, train_acc = exe.run(
                program=fluid.default_main_program(),                            #运行主程序
                feed=feeder.feed(data),                                          #喂入一个batch的数据
                fetch_list=[avg_cost, accuracy])                                 #fetch均方误差和准确率

            all_train_iter = all_train_iter+BATCH_SIZE
            all_train_iters.append(all_train_iter)
            all_train_costs.append(train_cost[0])
            all_train_accs.append(train_acc[0])


            if batch_id % 10 == 0:                                               #每10次batch打印一次训练、进行一次测试
                print("\nPass %d, Step %d, Cost %f, Acc %f" %
                      (pass_id, batch_id, train_cost[0], train_acc[0]))
        # 开始测试
        test_accs = []                                                            #测试的损失值
        test_costs = []                                                           #测试的准确率
        # 每训练一轮 进行一次测试
        for batch_id, data in enumerate(test_reader()):                           # 遍历test_reader
            test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # #运行测试主程序
                                          feed=feeder.feed(data),                #喂入一个batch的数据
                                          fetch_list=[avg_cost, accuracy])       #fetch均方误差、准确率
            test_accs.append(test_acc[0])                                        #记录每个batch的误差
            test_costs.append(test_cost[0])                                      #记录每个batch的准确率

        # 求测试结果的平均值
        test_cost = (sum(test_costs) / len(test_costs))                           # 每轮的平均误差
        test_acc = (sum(test_accs) / len(test_accs))                              # 每轮的平均准确率
        print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))


        # 如果保存路径不存在就创建
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        # 保存训练的模型，executor 把所有相关参数保存到 dirname 中
        fluid.io.save_inference_model(dirname=model_save_dir,
                                      feeded_var_names=["image"],
                                      target_vars=[predict],
                                      executor=exe)

    draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")

    print('训练模型保存完成！')