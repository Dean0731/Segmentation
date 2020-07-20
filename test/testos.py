import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json
import numpy as np
import matplotlib
font = {'family': 'MicroSoft YaHei','weight': 'bold','size': 10} # 中文乱码
matplotlib.rc("font", **font)
file = r'temp2.txt'
car = []
bus = []
van = []
others = []
X = []
with open(file,'r') as f:
    data = [json.loads(line) for line in f.readlines()]
for i,line in enumerate(data):
    X.append(i)
    temp = line.get(str(i))
    car.append(temp.get('car'))
    bus.append(temp.get('bus'))
    van.append(temp.get('van'))
    others.append(temp.get('others'))
X = np.divide(X,60)
# plt.figure(figsize=(9,6), dpi=500)
plt.plot()
plt.plot(X,car,label='car')
plt.plot(X,bus,label='bus')
plt.plot(X,van,label='van')
plt.plot(X,others,label='other')
plt.plot(X,np.sum([car,van,bus,others],axis=0),label='sum')
plt.legend(loc=2)
plt.xlabel('运行时间(分钟)')
plt.ylabel('数量')
plt.title('车辆流统计')

plt.show()