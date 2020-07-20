class A:
    def __init__(self):
        pass
x= A()
x.a = 12
a = x
x.a = 13
print(a.a)