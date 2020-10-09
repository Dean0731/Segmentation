import numpy as np
import util
path = r'C:\Users\root\Desktop\test\label.png'
# path = r'C:\Users\root\Desktop\label.png'
# path = r'C:\Users\root\Desktop\hha.jpg'


# arr = util.printImagePIL(path,True)
# util.printArray(arr)
arr = util.printImagecv2(path,True)
util.printArray(arr)