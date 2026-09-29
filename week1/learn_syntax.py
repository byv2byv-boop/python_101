"""
В этом файле будем изучать синтаксис языка программирования Пайтон.

"""

'''
Так тоже можно
'''
import math
import turtle

# Так мы будем оформлять комментарий
variable = 100

print (variable**2 + 13)
print (variable**2 + 13) 

print(f"Наш ответ номер 1: {variable**2+25}")
print("Наш ответ номер 1: ", variable**2+25) 

seconds_in_min = 60
mins_in_hour = 60
hours_in_day = 24
days_in_year = 365

seconds_in_year = seconds_in_min * mins_in_hour * hours_in_day * days_in_year

print(f"Количество секунд в году: {seconds_in_year}")

#Необычная арифметика Питона
our_number = 42
print(our_number**2)# числов в квадрате
print(our_number**3)# числов в кубе

print(our_number / 7)# числов делим на 7
print(our_number * 10)# числов в квадрате

print(our_number // 10 )# целочисленное деление: целая часть от деления без остатка
print(our_number % 10 )# остаток от деления на 10

our_number = our_number + 1

our_number += 2

our_sine = round(math.sin(math.pi/2),6)
our_cosine = round(math.cos(math.pi/2),6)

print (f"Наш родной синус: {our_sine}")
print (f"Наш родной косинус: {our_cosine}")
print (math.pi)

#поговорим про типы данных
my_name = "John jonson"
my_age = 42
my_city = "Moscow"
my_wine = 2.5

first_row_group = ["Максим","Даша", "Критсина", "Ксюша", "ВАня"]
second_row_group = [
    "Олег",
    "Сергей",
    "Сергей",
    "Радмила", 
    "Саша", 
    "Надя"
    ]

type(my_name)
type(my_age)
type(my_city)
type(my_wine)

type(first_row_group)
type(second_row_group)

my_secret_system = {
    0: "выходи",
    1: "Не выходи",
    2: "Пей", 
    3: "Читай книгу",
    4: "Спи",
    5: "учись",
}

type (my_secret_system)

print (my_secret_system[0])
print (my_secret_system[3])


#test imeni
student_name = "Ясельский Алексей Анатольевич"
smth = student_name.split()
print (smth [0])
print (smth [1])
print (smth [2])
print (len(smth))
other_list = smth[:2]
other_list = smth[-:2]

#Дома усвоить что такое список или словарь

print ("hellow world")
