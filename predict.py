import util
import logging
def main():
    logging.basicConfig(level=logging.INFO)
    type = util.flag.get("type")
    try:
        if type == 'tensorflow':
            from tf import predict
            predict.main()
        elif type == 'paddlepaddle':
            from paddlepaddle import predict
            # predict.main()
        else:
            from pytorch import predict
            predict.main()
            logging.info("默认执行pytorch")
    except Exception as e:
        logging.exception(e)
        util.sendMessageWeChat(e)
if __name__ == '__main__':
    main()