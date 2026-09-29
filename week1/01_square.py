import sys
import os

if sys.platform == "darwin":  # Проверка, что это macOS
    os.system('''osascript -e 'tell application "System Events" to set frontmost of every process whose unix id is '`echo $$`' to true' &''')
"""
01. Квадрат — первая программа.

Что показываем:
- черепаха умеет только ходить вперёд и поворачивать;
- четыре одинаковых шага = квадрат;
- вместо четырёх копий одних и тех же строк можно написать цикл for:
  компьютер повторяет за нас.
"""
import turtle

screen = turtle.Screen()
screen.setup(600, 600)
screen.title("01: квадрат")
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)
t = turtle.Turtle()
t.shape("turtle")

# Способ 1: «вручную» — четыре раза одно и то же
t.forward(150)
t.left(90)
t.forward(150)
t.left(90)
t.forward(150)
t.left(90)
t.forward(150)
t.left(90)

# Способ 2: то же самое, но циклом. Красный квадрат поменьше.
t.color("red")
for _ in range(4):
    t.forward(100)
    t.left(90)

# Попробуй: поменять 90 на 120 или на 60 — что получится?
screen.exitonclick()   # клик по окну закрывает его
