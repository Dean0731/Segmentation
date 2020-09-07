# @Time     : 2020/8/30 22:43
# @File     : generate_data_txt
# @Email    : dean0731@qq.com
# @Software : PyCharm
# @Desc     :
# @History  :
#   2020/8/30 Dean First Release
import os
def generate(src):
    with open(os.path.join(src,'data.txt'),'w',encoding='utf-8') as f:
        for root, dirs, files in os.walk(src, topdown=False):
            for name in files:
                if not 'Target' in root:
                    f.write("{};{}\n".format(os.path.join(root,name),os.path.join(root.replace('Input images','Target maps'),name)))
if __name__ == '__main__':
    src = r'G:\AI_dataset\massachusetts\Massachusetts576'
    generate(src)