import os
from PIL import Image

TILE = 192
COLUMNS = 40

files = []
for name in sorted(os.listdir("week2/movies/posters")):
    if name.endswith(".jpg"):
        files.append(name)


rows = len(files) // COLUMNS
if len(files) % COLUMNS != 0:
    rows += 1

mosaic = Image.new("RGB", (COLUMNS * TILE, rows * TILE), "black")
for i, name in enumerate(files):
    poster = Image.open("week2/movies/posters/" + name).resize((TILE, TILE))
    column = i % COLUMNS
    row = i // COLUMNS
    mosaic.paste(poster, (column * TILE, row * TILE))

mosaic.save("week2/movies/posters_mosaic.jpg", quality=90)
