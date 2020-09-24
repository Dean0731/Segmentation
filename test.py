import util
import logging
logging.basicConfig(level=logging.DEBUG)
@util.cls.Decorator.sendMessageDingTalk(util.flag.get("txt"))
@util.cls.Decorator.timer()
def test():
    pass
if __name__ == '__main__':
    test()

from util import func