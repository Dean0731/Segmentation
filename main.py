# @Time     : 2020/8/26 13:40
# @File     : main
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release



# from pytorch.example import pytorch_for_cnn
# pytorch_for_cnn.main()


import util
import logging
logging.basicConfig(level=logging.INFO)
dic = util.getCmdDict()
type = dic.get("type")
if type == 'tf':
    from tf import train
    train.main()
elif type == 'torch':
    from pytorch import train
    train.main()
elif type == 'paddlepaddle':
    from paddlepaddle import train
    train.main()
else:
    print("启动出错")

# python main.py --type paddlepaddle
# python main.py --type tf
# python main.py --type torch