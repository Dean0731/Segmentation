import torch
dic={"bg":0,"house":1}
class Segmentation():
    @staticmethod
    def getAcc(y_pred,y):
        y_pred = y_pred.argmax(dim=1)
        acc = 0.0
        for y_,y_pred_ in zip(y,y_pred):
            TP = torch.sum((y_==dic['house']) & (y_pred_==dic['house']))
            FP = torch.sum((y_==dic['bg']) & (y_pred_==dic['house']))
            FN = torch.sum((y_==dic['house']) & (y_pred_==dic['bg']))
            TN = torch.sum((y_==dic['bg']) & (y_pred_==dic['bg']))
            #acc = acc + float((TP+TN)/(TP+FN+FP+TN))
            acc = acc + torch.true_divide((TP+TN),(TP+FN+FP+TN))
        return acc
    # @staticmethod
    # def getIou(y_pred,y):
    #     y_pred = y_pred.argmax(dim=1)
    #     iou = 0
    #     for y_,y_pred_ in zip(y,y_pred):
    #         a = y_.sum()
    #         b = y_pred_.sum()
    #         y_pred_[y_pred_==0]=2
    #         c = y_.eq(y_pred_).sum()
    #         iou = iou + torch.true_divide(c,(a+b-c))
    #     return iou
    @staticmethod
    def getIou(y_pred,y):
        y_pred = y_pred.argmax(dim=1)
        iou = 0.0
        for y_,y_pred_ in zip(y,y_pred):
            TP = torch.sum((y_==dic['house']) & (y_pred_==dic['house']))
            FP = torch.sum((y_==dic['bg']) & (y_pred_==dic['house']))
            FN = torch.sum((y_==dic['house']) & (y_pred_==dic['bg']))
            iou = iou + float(TP/(TP+FN+FP))
        return iou
    @staticmethod
    def getRecall(y_pred,y):
        y_pred = y_pred.argmax(dim=1)
        recall = 0.0
        for y_,y_pred_ in zip(y,y_pred):
            TP = torch.sum((y_==dic['house']) & (y_pred_==dic['house']))
            FP = torch.sum((y_==dic['bg']) & (y_pred_==dic['house']))
            FN = torch.sum((y_==dic['house']) & (y_pred_==dic['bg']))
            recall = recall + float(TP/(TP+FN))
        return recall
    @staticmethod
    def getPrecision(y_pred,y):
        y_pred = y_pred.argmax(dim=1)
        precisoin = 0.0
        for y_,y_pred_ in zip(y,y_pred):
            TP = torch.sum((y_==dic['house']) & (y_pred_==dic['house']))
            FP = torch.sum((y_==dic['bg']) & (y_pred_==dic['house']))
            FN = torch.sum((y_==dic['house']) & (y_pred_==dic['bg']))
            precisoin = precisoin + float(TP/(TP+FP))
        return precisoin
