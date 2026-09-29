"""
21. Круги в квадратах — сетка со случайной палитрой (продвинутый уровень).

По присланному образцу: 5 x 5 квадратов, в каждом — круг другого цвета.
Цвета не любые, а из фиксированной палитры; случайность выбирает только,
какие два из них встретятся. Так работают многие цветные серии Мольнар и
«конкретных» художников: правило строгое, выбор случайный.

Главный приём: random.sample(palette, 2) даёт ДВА РАЗНЫХ цвета сразу.
"""
import turtle
import random

screen = turtle.Screen()
screen.setup(720, 720)
screen.bgcolor("white")
screen.title("21: круги в квадратах")
screen.tracer(5)

rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)

t = turtle.Turtle()
t.hideturtle()
t.penup()

palette = ["#f7d117", "#e8731a", "#e30f1f", "#c45ea3", "#ec6fab",
           "#1d7fc4", "#4c4a99", "#008d4b", "#7d4b3d", "#3b3b40"]

N = 5
CELL = 136          # шаг сетки
SQUARE = 124        # квадрат чуть меньше шага — остаётся белый зазор
START = -N * CELL / 2 + CELL / 2

for row in range(N):
    for col in range(N):
        cx = START + col * CELL
        cy = START + row * CELL
        background, dot = random.sample(palette, 2)     # два разных цвета

        # квадрат
        t.goto(cx - SQUARE / 2, cy - SQUARE / 2)
        t.fillcolor(background)
        t.begin_fill()
        for _ in range(4):
            t.forward(SQUARE)
            t.left(90)
        t.end_fill()

        # круг: черепаха встаёт под центром, circle рисует вокруг
        t.goto(cx, cy - SQUARE / 2 + 6)
        t.setheading(0)
        t.fillcolor(dot)
        t.begin_fill()
        t.circle(SQUARE / 2 - 6, steps=360)
        t.end_fill()

screen.update()
# Попробуй: свою палитру из 3 цветов; N = 10; круг radius = SQUARE / 3 (маленькие точки);
# random.seed(7) в начале — «случайность» станет повторяемой.
screen.exitonclick()
