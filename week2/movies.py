import json
import os
import requests

with open("data/tspdt_movies.json", encoding="utf-8") as file:
    films = json.load(file)

# print(f"Type films: {type(films)}")
# for key in films:
#     print(key)

films_list = films["movies"]
# print(f"Type films_list: {type(films_list)}")
# print(f"ПЕРВЫЙ ЭЛЕМЕНТЫ НАШЕГО СПИСКА: {films_list[0]}")
# print(f"Первый фильм, это тип данных: {type(films_list[0])}")

#---------------------------------------
os.makedirs("week2/movies/posters", exist_ok=True)

def file_name(film):
    """0001_citizen_kane_1941.jpg"""
    title = ""
    for char in film["title_clean"].lower():
        if char.isalnum():
            title += char
        elif char == " ":
            title += "_"
    rank = str(film["rank"]).zfill(4) #zfill помогает сортировать 1---> 0001

    return rank + "_" + title + "_" + str(film["year"])+".jpg"

first_film = films_list[667]
correct_name = file_name(first_film)

print(correct_name)

for film in films_list:
    path = "week2/movies/posters/" + file_name(film)
    if os.path.exists(path):
        continue

    response = requests.get(film["image_url"])
    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
        print(path)
    else:
        print(f"НЕУДАЧА! {path}, {response.status_code}")


print("Hello world!")