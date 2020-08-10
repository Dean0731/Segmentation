# @Time     : 2020/8/10 17:45
# @File     : dataset_tools
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/10 Dean First Release
from util.dataset.AerialImage import AerialImage
from util.dataset.CountrySide import CountrySide
from util.dataset.Massachusetts import Massachusetts
def selectDataset(str='A',data_size='tif_576',parent="/home/dean/"):
    if str == 'A':
        dataset = AerialImage(parent= parent,data_size=data_size)
    elif str == 'C':
        dataset = CountrySide(parent= parent,data_size=data_size)
    elif str == 'M':
        dataset = Massachusetts(parent= parent,data_size=data_size)
    else:
        print("错误:未找到数据集.")
    return dataset