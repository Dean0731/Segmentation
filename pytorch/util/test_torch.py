# @Time     : 2020/8/25 17:40
# @File     : test_torch
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 2
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())

device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())




print("--------------------------------")
available = torch.cuda.is_available()
print("cuda available:",available)
gpu_number = torch.cuda.device_count()
print("cuda count:",gpu_number)
for i in range(gpu_number):
    print(torch.cuda.device(i),torch.cuda.get_device_name(i))
print(torch.device("cpu"))

