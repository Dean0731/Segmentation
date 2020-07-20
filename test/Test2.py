import demjson
import json
file = r'temp.txt'   # 原始文件路径
file2 = r'temp2.txt' # 保存文件路径
sum = 37324     # 数据总行数
avg_num = 24  # 每几条一个平均

with open(file, 'r', encoding="utf-8") as f:
    data = [demjson.decode(line)for line in f.readlines()]

car,bus,van,others = 0,0,0,0
with open(file2,'w',encoding='utf-8')as f:
    for i in range(sum):
        temp = data[i].get(i+1)
        car = car + temp.get('car')
        bus = bus + temp.get('bus')
        van = van + temp.get('van')
        others = others + temp.get('others')
        if i % avg_num == avg_num -1 or i == sum-1:
            num = i // avg_num
            if i == sum - 1:
                avg_num = sum % avg_num
            avg_car = int(car / avg_num)
            avg_bus = int(bus / avg_num)
            avg_van = int(van / avg_num)
            avg_others = int(others / avg_num)
            dic = {num:{'car':avg_car,'bus':avg_bus,'van':avg_van,'others':avg_others}}
            f.write(json.dumps(dic)+"\n")
            print(dic)
            car,bus,van,others = 0,0,0,0




