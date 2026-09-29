#Разбираемся в том что такое функция: "возможность группироать код в маленький блок"
import turtle

screen = turtle.Screen()
screen.setup(600,600)
rootwindow = screen.getcanvas().winfo_toplevel()
rootwindow.attributes('-topmost', True)
screen.title("Прыгающий мяч")
screen.tracer(0) #видимость анимации"

#инициализация черепахи
ball = turtle.Turtle() #создает новый графический объект (черепашку) и присваивает ему имя ball.
ball.shape("circle") #меняет форму объекта со стандартной стрелочки на круг.
ball.shapesize(2) #увеличивает размер круга в 2 раза по сравнению со стандартным размером.
ball.color("blue") #окрашивает круг в синий цвет.
#ball.penup()


LIMIT = 260
velocity = [2,15]

def madball():
    x, y = ball.position()
    x = x + velocity[0]
    y = y + velocity[1]

    if x > LIMIT or x < -LIMIT:
        velocity[0] = -velocity[0]
    if y > LIMIT or y < -LIMIT:
        velocity[1] = -velocity[1]

    ball.goto(x,y)
    screen.update()
    screen.ontimer(madball,25) #вызов функции через 16 м/сек - это примерно соответсвует 24fps / 


madball()
screen.mainloop()
# показывал проект cory arcangel - circed banding Super Mario Clouds - lowfi estetic, 
# один из свамых молодых и востребованных художников но сайт у него в css (стилево)
# 386dx - караоке Алексей Анатольевич Шульгин смотри на "Дом культуры" в Ютубе
# Антон Москелиада один из крутейшихз преподавателей

