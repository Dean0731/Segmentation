logs = {"step":1,"loss":2,"acc":3,"iou":4}

for key,value in {k:v for (k,v) in logs.items() if k not in["step","loss"]}.items():
    print(key,value)