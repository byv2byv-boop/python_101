"""
02. Многоугольники — одна формула вместо шести программ.

Что показываем:
- переменная n: число сторон записано ОДИН раз, а используется везде;
- формула 360 / n: чтобы обойти фигуру, черепаха в сумме поворачивается на 360°;
- внешний цикл перебирает n = 3, 4, 5... и рисует фигуры рядом.
"""
import turtle

screen = turtle.Screen()
screen.setup(900, 400)
screen.title("02: многоугольники")
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)

t = turtle.Turtle()
t.speed(0)              # 0 — максимальная скорость
t.penup()
t.goto(-420, -50)
t.pendown()

for n in range(3, 9):           # n = 3, 4, 5, 6, 7, 8
    angle = 360 / n             # угол поворота зависит от числа сторон
    for _ in range(n):
        t.forward(45)
        t.left(angle)
    # сдвигаемся правее — там будет следующая фигура
    t.penup()
    t.forward(140)
    t.pendown()

# Попробуй: range(3, 40) и forward(8) — многоугольник превращается в круг.
screen.exitonclick()
