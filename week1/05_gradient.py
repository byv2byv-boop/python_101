import turtle
from random import random

CANVAS_W = 1000
CANWAS_H = 1000

#Инициализайия экрана
screen = turtle.Screen()
screen.setup(CANVAS_W, CANWAS_H)
screen.tracer(2)
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)

#Инициализайия черепахи
t = turtle.Turtle()
t.hideturtle()
t.speed(0)

#КОл-во рядов, начальная координата, кол-во ячеек
num_rows = 20
num_cols = 20
start_x = -CANVAS_W / 2
start_y= CANWAS_H / 2

cell_w = CANVAS_W / num_cols
cell_h = CANWAS_H / num_rows

#2 цвета
color_a = (0.95, 0.65, 0.80)
color_b = (0.40, 0.75, 0.95)

# Функция которая позволет нам скользить по линии от а до б, регулируя коэффициент К. 
# К от 0 до 1. Если К = 0 то остаемся в точке а, 
# если К=1 то оказываемся в б, Если ка = 0,5 - на середине
def mix(a,b,k):
    return a + (b - a)*k

#Функция которая берет 2 цвет а и с учетом параметра К
# ИМы оказываемся в другом цветею Если к = 0б то мы в цвете 1
# К=1 то мы в цете 2
# К = 0,5 посередине между двумя цветами
def between(color1,color2,k):
    r = mix(color1[0], color2[0], k)
    g = mix(color1[1], color2[1], k)
    b = mix(color1[2], color2[2], k)
    return (r, g, b)


#Функция доя ячейки
def cell(x, y, color):
    t.penup()
    t.goto(x,y)
    t.pendown()
    t.fillcolor(color)
    t.pencolor(color)

    t.begin_fill()
    for i in range(2): #эта стока (синтаксис) подходит если рисуем прямоугольники а не квадраты
        t.forward(cell_w)
        t.right(90)
        t.forward(cell_h)
        t.right(90)
    t.end_fill()

num_cells = num_cols * num_rows

for i in range(num_cells):
    row = i // num_cols
    col = i % num_cols

    #k = (row + col) % 2 #квадратики
    #k = col/(num_cols - 1) #градиент горизонтальный
    k = row / (num_rows - 1) #градиент вертикальный
    #k = (row + col) / (num_rows + num_cols - 2) #градиент диагональный

    x = start_x + col *cell_w
    y = start_y - row * cell_h

    color = between(color_a, color_b, k)
    cell(x, y, color)

screen.update()
screen.exitonclick()



