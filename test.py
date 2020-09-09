from random import randint



path = r'C:\Users\root\Desktop\1_json\test.png'



#


import os
from PIL import Image
def randomPalette(length, min, max):
    return [randint(min, max) for x in range(length)]
src=r'C:\Users\root\Desktop\1_json'
k = 1
for root, dirs, files in os.walk(src, topdown=False):
    for name in files:
        path = os.path.join(root,name)
        if '22828930' in path:
            img = Image.open(path)
            img = img.convert('P')
            i = randomPalette(0, 0, 0)
            img.putpalette(i)
            img.save(path)
            print(i,path)
            k = k+1

from PIL import Image
img = Image.open(r'C:\Users\root\Desktop\1_json\22828930_15_000.tif.png')

import numpy as np
img = np.asarray(img)
for i in range(300):
    for j in range(200):
        print(img[i][j],end='')
    print()