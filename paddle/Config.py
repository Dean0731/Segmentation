import os
import json

# 设置要生成文件的路径
data_root_path = '/home/aistudio/images/face'
# 所有类别的信息
class_detail = []
# 获取所有类别保存的文件夹名称，这里是['zhangziyi', 'jiangwen', 'pengyuyan']
class_dirs = os.listdir(data_root_path)
# 类别标签
class_label_dict = {'zhangziyi': 0, 'jiangwen': 1, 'pengyuyan': 2,'dzf':3}
# 获取总类别的名称
father_paths = data_root_path.split('/')    #['', 'home', 'aistudio', 'images', 'face']
while True:
    if father_paths[father_paths.__len__() - 1] == '':
        del father_paths[father_paths.__len__() - 1]
    else:
        break
father_path = father_paths[father_paths.__len__() - 1]
# 把生产的数据列表都放在自己的总类别文件夹中
data_list_path = '/home/aistudio/%s/' % father_path
# 如果不存在这个文件夹,就创建
isexist = os.path.exists(data_list_path)
if not isexist:
    os.makedirs(data_list_path)
# 清空原来的数据
with open(data_list_path + "test.list", 'w') as f:
    pass
with open(data_list_path + "trainer.list", 'w') as f:
    pass
# 总的图像数量
all_class_images = 0
# 读取每个类别
for class_dir in class_dirs:
    # 每个类别的信息
    class_detail_list = {}
    test_sum = 0
    trainer_sum = 0
    # 统计每个类别有多少张图片
    class_sum = 0
    # 获取类别路径
    path = data_root_path + "/" + class_dir
    # 获取所有图片
    img_paths = os.listdir(path)

    for img_path in img_paths:                                  # 遍历文件夹下的每个图片
        name_path = path + '/' + img_path                       # 每张图片的路径
        if class_sum % 10 == 0:                                 # 每10张图片取一个做测试数据
            test_sum += 1                                       #test_sum测试数据的数目
            with open(data_list_path + "test.list", 'a') as f:
                f.write(name_path + "\t%d" % class_label_dict[class_dir] + "\n") #class_label 标签：0,1,2
        else:
            trainer_sum += 1                                    #trainer_sum测试数据的数目
            with open(data_list_path + "trainer.list", 'a') as f:
                f.write(name_path + "\t%d" % class_label_dict[class_dir] + "\n")#class_label 标签：0,1,2
        class_sum += 1                                          #每类图片的数目
        all_class_images += 1                                   #所有类图片的数目

    # 说明的json文件的class_detail数据
    class_detail_list['class_name'] = class_dir             #类别名称，如jiangwen
    class_detail_list['class_label'] = class_label_dict[class_dir]          #类别标签，0,1,2
    class_detail_list['class_test_images'] = test_sum       #该类数据的测试集数目
    class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
    class_detail.append(class_detail_list)

# 获取类别数量
all_class_sum = class_dirs.__len__()
# 说明的json文件信息
readjson = {}
readjson['all_class_name'] = father_path                  #文件父目录
readjson['all_class_sum'] = all_class_sum                #
readjson['all_class_images'] = all_class_images
readjson['class_detail'] = class_detail
jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
with open(data_list_path + "readme.json",'w') as f:
    f.write(jsons)
print ('生成数据列表完成！')
print ("标签及其类别：{}".format(class_label_dict))

