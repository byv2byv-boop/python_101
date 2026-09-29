from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 4320, 7680 # ГЛОБАЛЬНЫЕ ВЕЩИ ЗАДАЕМ ЗАГЛАВНЫМИ БУКВАМИ
MARGIN = 260
LINE_HEIGHT = 21
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf",16) #Подтягиваем шрифт

with open ("data/onegin.txt", encoding="utf-8") as file:
    words = file.read().split() # прочитали файл и сразу ему сделали сплит. разбиваем по всевозможным параметрам

print (f"Кол-во слов: {len(words)}")


""" есть ширина 4320 , есть рамка марджин. берем первое слово,
 считаем ширину в пикселях и потихоньку наращиваем новыми словами, в какой-то момент не поместится новое слово,
 и мы начнем новую строку"""


lines = [] # неограниченное количество строк
line= words [0] # первая линия — это первое слово из списка

for word in words [1:]: # для каждого слова из списка начиная с того, что имеет индекс 1
    longer = line + " " + word # переменная лонгер содержит в себе слово, пробел и еще одно слово
    if FONT.getlength (longer) <= WIDTH -2 * MARGIN: # если длина шрифта больше, чем рабочее поле
        line = longer #переменной лайн присвоили значение лонгер
    else:
        lines.append (line) #строка полная — слово идет в новую строку
        line = word #начинаем новую строку с этого слова

lines.append (line)
print (f"Кол-во строк: {len (lines)}")

image = Image.new("RGB", (WIDTH, HEIGHT), "#ffffff")
draw = ImageDraw.Draw (image)

#################################

evgenis = []
tatianas = []

#################################333

y=MARGIN

for line in lines:
    draw.text ((MARGIN, y), line, font = FONT, fill = "#3c3a34") 
    #########################

    x = MARGIN
    for word in line.split (" "):
        if "Евген" in word or "Онеги" in word:
            evgenis.append  ((x, y + LINE_HEIGHT))  # добавление элемента в конец списка
        if "Татьян" in word or "Тан" in word:
            tatianas.append ((x,y + LINE_HEIGHT))
        x += FONT.getlength (word + " ")

    ###################

    y = y+LINE_HEIGHT

########################################

print (f"Количество Евгениев: {len(evgenis)} . Количество Татьян: {len(tatianas)}")

pen = ImageDraw.Draw(image, "RGBA")

for evgeny in evgenis: # соединим каждого Евгения с каждой Татьяной
    for tatiana in tatianas:
        pen.line ([evgeny, tatiana], fill = (190,40,40,40), width =3)

##############################

image.save ("onegin_poster.png")
print ("ВСЁ ГОТОВО!!")


print ("////")