# @Time     : 2020/8/26 13:40
# @log     : main
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release


# from pytorch.example import pytorch_for_cnn
# pytorch_for_cnn.main()
# --epochs 2 训练轮数
# --info 1/0 是否发送消息
# --log 是否有日志
# --type 框架选择
import sys
sys.path.append(r'/public/home/huzhanpeng/anaconda3/envs/python3.6/lib/python3.6/site-packages')
import util
import logging
import os

@util.cls.Decorator.sendMessageWeChat(flag=int(util.flag.get("info")))
@util.cls.Decorator.timer(flag=True)
def main():
    logging.basicConfig(level=logging.ERROR)
    type = util.flag.get("type")
    log = util.flag.get('log')
    if log != None:
        if not os.path.exists(str(log)):
            util.flag['log'] = util.func.get_dir(os.path.join(util.getParentDir(), 'source/pytorch'))
        print("log_dir:",util.flag['log'])
    try:
        if type == 'tensorflow':
            from tf import train
            train.main()
        elif type == 'paddlepaddle':
            from paddlepaddle import train
            train.main()
        else:
            from pytorch import train
            logging.info("默认执行pytorch")
            train.main()

    except Exception as e:
        logging.exception(e)
        if int(util.flag.get("info")):
            util.sendMessageWeChat(e)

if __name__ == '__main__':
    main()
