import numpy as np

a = np.array([0,0,0,0,1,1,1,1,1,0,0,0,0])
b = np.array([0,0,0,0,0,0,1,1,1,1,1,1,0])
all = np.sum(np.equal(a,b))
TP = np.sum((a==1) & (b==1))
FP = np.sum((a==0) & (b==1))
FN = np.sum((a==1) & (b==0))

print(TP,FP,FN)

import pytorch.util.Config