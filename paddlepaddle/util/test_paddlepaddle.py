# @Time     : 2020/8/25 17:40
# @File     : test_torch
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/25 Dean First Release
import paddle.fluid
flag = paddle.fluid.install_check.run_check()
print(flag)