# # class Odam:
# #     def __init__(self,name,surname,age):
# #         self.name=name
# #         self.surname=surname
# #         self.age=age
# #
# #     def Add(self):
# #         a, b = int(input("Son kiriting:")), int(input("Keyingi sonni kiriting:"))
# #         return a + b
# #
# # odam=Odam('Davron','Aliqulov',2001)
# # print(odam.Add())
#
#
# class Car:
#
#   def __init__(self, make, model, year, color, number):
#     self.make = make
#     self.model = model
#     self.year = year
#     self.color = color
#     self.number = number
#
#   def get_description(self):
#     return f"{self.year} {self.make} {self.model}, {self.color} rangda,raqami {self.number} ."
#
# # make =input("Modelni kiriting:")
# #     self.model = model
# #     self.year = year
# #     self.color = color
# #     self.number = number
# my_car = Car("Chevrolet", "Spark", 2020, "oq","70B234AA")
# print(my_car.get_description())
#
class Triangle:
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c
    def perimetr(self):
        return self.a+self.b+self.c
    def area(self):
        return (self.perimetr()/2 * (self.perimetr()/2-self.a)* (self.perimetr()/2-self.b)* (self.perimetr()/2-self.c))**(1/2)
a=int(input("3<ning 1- tomonini kiriting:"))
b=int(input("3<ning 2- tomonini kiriting:"))
c=int(input("3<ning 3- tomonini kiriting:"))
triangle=Triangle(a,b,c)
print(triangle.perimetr(),triangle.area())