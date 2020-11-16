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

@util.cls.Decorator.sendMessageWeChat(flag=int(util.flag.get("info")))
@util.cls.Decorator.timer(flag=True)
def main():
    logging.basicConfig(level=logging.ERROR)
    type = util.flag.get("type")
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
