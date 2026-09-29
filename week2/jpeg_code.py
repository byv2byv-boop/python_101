# в этом файле мы прогоняем цикл пересохранений jpeg файла, уменьшая его качество с каждым разом

from PIL import Image

from random import randint

path_original = "data/1665_Girl_with_a_Pearl_Earring.jpg"
path_gen = "week_2/jpeg_code/output/1665_Girl_with_a_Pearl_Earring_gen.jpg"


img = Image.open(path_original)

for i in range(10000):
    print(f"Итерация #{i}")
    img.save(path_gen, quality = randint(70, 80))
    img = Image.open(path_gen)
    img.load()
