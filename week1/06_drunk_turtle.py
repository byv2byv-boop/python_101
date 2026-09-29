"""
06. Пьяная черепаха — случайность и условие с «или» (or).

Что показываем:
- random.choice(список) — компьютер сам выбирает случайный поворот;
- if с or: если ушла слишком влево ИЛИ вправо ИЛИ вверх ИЛИ вниз —
  возвращаем её домой и меняем цвет;
- x, y = t.position() — черепаха знает, где она.
"""
import turtle
import random

screen = turtle.Screen()
screen.setup(600, 600)
screen.title("06: пьяная черепаха")
screen.tracer(3)                     # показывать каждый 3-й шаг, иначе слишком медленно

t = turtle.Turtle()
t.shape("turtle")
t.speed(0)
t.width(2)

LIMIT = 250                          # дальше от центра уходить нельзя
turns = [-90, -45, 0, 45, 90]        # возможные повороты
colors = ["red", "blue", "green", "purple", "orange"]

for step in range(1200):
    t.left(random.choice(turns))
    t.forward(15)

    x, y = t.position()
    if x > LIMIT or x < -LIMIT or y > LIMIT or y < -LIMIT:
        t.penup()
        t.home()                     # заблудилась — телепортируем в центр
        t.pendown()
        t.pencolor(random.choice(colors))

screen.update()                      # дорисовать последние шаги
# Попробуй: turns = [-10, 0, 10] — походка станет плавной; LIMIT = 100.
screen.exitonclick()
