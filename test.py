import cv2

def write():
    img = cv2.imread(path)

    # cv2.rectangle(img,(100,50),(300,150),(0,0,255),-2)
    cv2.putText(img,world,(100,100),cv2.FONT_HERSHEY_COMPLEX,2.0,(100,200,200),5)

    cv2.rectangle(img,(black[0],black[1]),(black[2],black[3]),(0,0,0),2)
    cv2.rectangle(img,(red[0],red[1]),(red[2],red[3]),(0,0,255),2)
    cv2.rectangle(img,(green[0],green[1]),(green[2],green[3]),(0,255,0),2)
    cv2.rectangle(img,(blue[0],blue[1]),(blue[2],blue[3]),(255,0,0),2)
    cv2.imwrite(path.split(".")[0]+"-result.jpg",img)

if __name__ == '__main__':
    path = r'G:\car\000001.jpg'

    # 字符
    world = '#16'
    # [左上点x,左上点y,右下x，右下y]
    black=[300,300,500,500]
    red= [i+10 for i in black]
    green=[i+20 for i in black]
    blue=[i+30 for i in black]

    write()


