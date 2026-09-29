import os 
import numpy
from PIL import Image

FOLDER= "week2/movies/posters/" #относительная ссылка, не нужно полную прописывать (абсолютную) траекторию. считаем от нашей корневой папки проекта

files= os.listdir(FOLDER) #перечилсяет все файлы, которые находятся в папке? не обязательно по порядку берет
files = sorted (files) #Переменную перезаписали, отсортировав список
# files= sorted.(os.listdir(FOLDER)) #то же самое в одну строку

for file in files [:10]: # 10 раз вывести название файлов, которые мы получили из папки
    print (file)

files_jpg = []
for file in files:
    if ".jpg" in file:
        files_jpg.append(file)

files_jpg = [f for f in files if ".jpg" in f] 

#смешиваем два изображения
a= Image.open(FOLDER + files[8]).convert("RGB")
b= Image.open(FOLDER + files[7]).convert("RGB").resize(a.size) #по идее тут мы подогнали вторую картинку под размер первой
Image.blend(a,b, 0.5).save("week2/movies/blend_two_images.jpg")

total = numpy.zeros((a.height, a.width, 3))
for f in files_jpg:
    image = Image.open(FOLDER + f).convert("RGB").resize(a.size)
    total+= numpy.array(image)

    average = total / len(files_jpg)
    Image.fromarray(average.astype("uint8")).save("week2/movies/blend_all.jpg")

print ("______")