from WEEK1.our_functions import give_me_any_power, give_me_squared_number
from random import random, randint


for i in range(10):
    # print(i)
    print(f"Квадрат числа {i} = {i**2}")


    def give_me_squared_number(my_number):  #ive_me_squared_numbe - это название / (my_number) - параметр
        squared_number = my_number**2 #вся строка это тело функции
        return squared_number #выход


    def give_me_any_power (my_number, to_the_power):
        return my_number ** to_the_power

    my_var = give_me_squared_number(9)
    my_other_var = give_me_any_power (9,9)
    print(f"my_var = {my_var}")
    print(f"my_other_var = {my_other_var}")
