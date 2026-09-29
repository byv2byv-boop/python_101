"""
05. Цветная спираль — список цветов и снова %.

Что показываем:
- список: colors = [...] — несколько значений в одной переменной;
- colors[i % len(colors)] — берём цвета по кругу: 0, 1, 2, ..., 6, 0, 1, 2...
  Без остатка от деления пришлось бы писать отдельный if на каждый цвет;
- шаг растёт на каждом витке — получается спираль.
"""
import turtle

screen = turtle.Screen()
screen.setup(700, 700)
screen.bgcolor("black")
screen.title("05: цветная спираль")
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)

colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]

t = turtle.Turtle()
t.speed(0)
t.width(2)

for i in range(150):
    t.pencolor(colors[i % len(colors)])   # i идёт до 149, а индекс всегда 0..6
    t.forward(i * 2)                      # каждый шаг длиннее предыдущего
    t.left(59)                            # чуть меньше 60° — и фигура «поплыла»

t.hideturtle()
# Попробуй: left(90), left(91), left(121); свой список цветов из двух штук.
screen.exitonclick()
