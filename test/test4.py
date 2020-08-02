# from network import Mine_Segnet3
# from util import Tools
# model = Mine_Segnet3.Segnet(3072,3072,3)
# Tools.computerNetworkSize(model)
import requests
proxy = {
    'http': 'http://129.226.156.219:8118',
    'https': 'https://129.226.156.219:8118',
}
# ret = requests.get('https://www.baidu.com',proxies=proxy)
ret = requests.get('https://www.googel.com',proxies=proxy)
print(ret.content)