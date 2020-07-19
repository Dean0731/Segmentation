d = {'a':1,'b':2}
temp = 'a'
def a(str):
    return len(str)+100
d[temp] = a(temp)
print(d)