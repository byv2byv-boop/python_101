"""
03. Концентрические круги.

Что показываем:
- t.circle(r) рисует круг, но черепаха стоит НЕ в центре, а на его краю;
- чтобы круги были вложенными, перед каждым спускаемся вниз на радиус;
- range(20, 260, 20) — цикл с шагом: 20, 40, 60, ... 240.
"""
import turtle

screen = turtle.Screen()
screen.setup(600, 600)
screen.title("03: концентрические круги")
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)

t = turtle.Turtle()
# t.speed(0)
# t.hideturtle()

for r in range(20, 260, 20):
    t.penup()
    t.goto(0, -r)      # встаём под центром на расстоянии r
    t.pendown()
    t.circle(r)        # круг радиуса r, центр окажется в точке (0, 0)

# Попробуй: шаг 10 вместо 20; t.width(3); t.pencolor("red") для каждого второго
# круга — подсказка: if r % 40 == 0 (остаток от деления, см. пример 04).
screen.exitonclick()
