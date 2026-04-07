

class A():
    def method_a(self):
        print("Method A")

class B():
    def method_b(self):
        print("Method B")


class C:
    def __init__(self):
        print("Creating instance of C")
        a = A()
        return a
    
    def method_c(self):
        print("Method C")
    

c = C()
print(type(c))
c.method_a()
c.method_c()