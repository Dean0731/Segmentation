# from network import Mine_Segnet3
# from util import Tools
# model = Mine_Segnet3.Segnet(3072,3072,3)
# Tools.computerNetworkSize(model)
import requests
proxy = {
    'http': 'socks5://127.0.0.1:1080',
    'https': 'socks5://127.0.0.1:1080',
}
proxy = {
    'http': 'http://127.0.0.1:1080',
    'https': 'https://127.0.0.1:1080',
}
# ret = requests.get('https://www.baidu.com',proxies=proxy)
ret = requests.get('https://www.google.com/',proxies=proxy)
print(ret.content)