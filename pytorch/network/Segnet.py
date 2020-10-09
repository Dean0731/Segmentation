# @Time     : 2020/8/26 17:18
# @File     : Segnet
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/26 Dean First Release
from torch import nn
import torch


class Segnet(nn.Module):
    def __init__(self, input_num, label_num):
        super(Segnet, self).__init__()

        self.conv11 = nn.Conv2d(input_num, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, label_num, kernel_size=3, padding=1)

    def forward(self, x):
        x11 = nn.functional.relu(self.bn11(self.conv11(x)), inplace=True)
        x12 = nn.functional.relu(self.bn12(self.conv12(x11)), inplace=True)
        x1p, id1 = nn.functional.max_pool2d(x12,
                                            kernel_size=2,
                                            stride=2,
                                            return_indices=True)

        x21 = nn.functional.relu(self.bn21(self.conv21(x1p)), inplace=True)
        x22 = nn.functional.relu(self.bn22(self.conv22(x21)), inplace=True)
        x2p, id2 = nn.functional.max_pool2d(x22,
                                            kernel_size=2,
                                            stride=2,
                                            return_indices=True)

        x31 = nn.functional.relu(self.bn31(self.conv31(x2p)), inplace=True)
        x32 = nn.functional.relu(self.bn32(self.conv32(x31)), inplace=True)
        x33 = nn.functional.relu(self.bn33(self.conv33(x32)), inplace=True)
        x3p, id3 = nn.functional.max_pool2d(x33,
                                            kernel_size=2,
                                            stride=2,
                                            return_indices=True)

        x41 = nn.functional.relu(self.bn41(self.conv41(x3p)), inplace=True)
        x42 = nn.functional.relu(self.bn42(self.conv42(x41)), inplace=True)
        x43 = nn.functional.relu(self.bn43(self.conv43(x42)), inplace=True)
        x4p, id4 = nn.functional.max_pool2d(x43,
                                            kernel_size=2,
                                            stride=2,
                                            return_indices=True)

        x51 = nn.functional.relu(self.bn51(self.conv51(x4p)), inplace=True)
        x52 = nn.functional.relu(self.bn52(self.conv52(x51)), inplace=True)
        x53 = nn.functional.relu(self.bn53(self.conv53(x52)), inplace=True)
        x5p, id5 = nn.functional.max_pool2d(x53,
                                            kernel_size=2,
                                            stride=2,
                                            return_indices=True)

        x5d = nn.functional.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = nn.functional.relu(self.bn53d(self.conv53d(x5d)), inplace=True)
        x52d = nn.functional.relu(self.bn52d(self.conv52d(x53d)), inplace=True)
        x51d = nn.functional.relu(self.bn51d(self.conv51d(x52d)), inplace=True)

        x4d = nn.functional.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = nn.functional.relu(self.bn43d(self.conv43d(x4d)), inplace=True)
        x42d = nn.functional.relu(self.bn42d(self.conv42d(x43d)), inplace=True)
        x41d = nn.functional.relu(self.bn41d(self.conv41d(x42d)), inplace=True)

        x3d = nn.functional.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = nn.functional.relu(self.bn33d(self.conv33d(x3d)), inplace=True)
        x32d = nn.functional.relu(self.bn32d(self.conv32d(x33d)), inplace=True)
        x31d = nn.functional.relu(self.bn31d(self.conv31d(x32d)), inplace=True)

        x2d = nn.functional.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = nn.functional.relu(self.bn22d(self.conv22d(x2d)), inplace=True)
        x21d = nn.functional.relu(self.bn21d(self.conv21d(x22d)), inplace=True)

        x1d = nn.functional.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = nn.functional.relu(self.bn12d(self.conv12d(x1d)), inplace=True)
        x11d = self.conv11d(x12d)

        x = x11d
        return x