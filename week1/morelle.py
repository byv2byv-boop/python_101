import turtle
from random import random, randint, choice

colours = ["red","green","blue"]

for index in range(10):
    #random_number = random()
    random_number = randint(13,379)
    random_paragr = randint(1,5)
    random_color = choice(colours)
    
    print(f"""
Генерация наше случайная страница: {random_number}. 
наш случайный апраграф {random_paragr}. 
наш случайный цвет {random_color}
""")

print("---"*10)
#настраиваем окно черепахи
screen = turtle.Screen()
screen.setup(1000,1000)
#rootwindow = screen.getcanvas().winfo_toplevel()
#rootwindow.attributes('-topmost', True)
screen.bgcolor("#315F9C")
screen.tracer(0)


#настраиваем черепаху
t=turtle.Turtle()
t.hideturtle()
t.speed(0)
t.color("#C44743")

size = 5 #мы заранее посчитали размер пиксельного квадрата тдля экрана"

for row in range(200):
    for col in range(200):
        x = -500 + col * size
        y = 500 - row * size

        if random() < 0.5:
            t.penup()
            t.goto(x,y)
            t.pendown()

            t.begin_fill()

            for _ in range(4):
                t.forward(size)
                t.right(90)

            t.end_fill()

        # screen.update()

screen.update()
screen.exitonclick()


print("Привет миир")