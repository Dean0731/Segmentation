# @Time     : 2020/8/25 17:45
# @File     : Import
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import sys
import os
parent = os.path.dirname(sys.path[0])
grandfater = os.path.dirname(parent)
sys.path.append(grandfater)
def a():
    pass