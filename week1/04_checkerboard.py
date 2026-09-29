"""
04. Шахматная доска — зачем нужен остаток от деления (%).

Что показываем:
- вложенные циклы: для каждой строки — для каждой клетки в ней;
- (row + col) % 2 — остаток от деления на 2: получается 0 или 1,
  и они чередуются ровно как клетки шахматной доски;
- if / else: если остаток 0 — чёрная клетка, иначе — белая.
"""
import turtle

screen = turtle.Screen()
screen.setup(600, 600)
screen.title("04: шахматная доска")
screen.tracer(1)        # рисуем всё сразу, без анимации черепахи
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)
t = turtle.Turtle()
t.hideturtle()

CELL = 50               # размер клетки в пикселях
N = 8                   # клеток в ряду
start = -N * CELL / 2   # координата левого нижнего угла доски

for row in range(N):
    for col in range(N):
        # сумма номеров строки и столбца то чётная, то нечётная
        if (row + col) % 2 == 0:
            t.fillcolor("black")
        else:
            t.fillcolor("white")

        t.penup()
        t.goto(start + col * CELL, start + row * CELL)
        t.pendown()
        t.begin_fill()
        for _ in range(4):
            t.forward(CELL)
            t.left(90)
        t.end_fill()

screen.update()

# Та же доска в консоли — только из остатков 0 и 1:
for row in range(N):
    line = ""
    for col in range(N):
        line = line + str((row + col) % 2) + " "
    print(line)

# Попробуй: заменить % 2 на % 3 — будет ли это доска?
# А если писать просто col % 2, без row?
screen.exitonclick()
