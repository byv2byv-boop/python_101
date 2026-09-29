#Знакомиммся с условными операторами и логическими конструкциями, функция ми, циклами








#Самый простой цикл
for i in range(13):
    print(i)

for a in range(15):
    print(a**2)

for cube in range(16):
    print(cube**3)

print('______________________________')

for kristina in range (65, 77, 2):
    print(f"Счаствилое число:{kristina}")

print('______________________________')

color = [
    'red',
    'green',
    'blue',
    'yves_klein_blue'
]

print (f"our colours are: {color}")
for fav_color in color: 
    print(f"current colur from {fav_color}")


idxs = [0,1,2,3,4,5,6,7]
for my_idx in idxs:
    print(f"My current idx: {my_idx}.В квадрате {my_idx**2}")    

slice_idx = idxs[2:5]
print(slice_idx)

slice_w_step = idxs[1:7:2]
print(slice_w_step)

reverse_idxs = idxs [::-1]
print(reverse_idxs)

for element in reverse_idxs:
    print(element)
print('--'*30)
for bang in range(100):
    ostatok = bang % 4
    if bang % 4 == 0:
        print("BANG!")
    else:
        print("---")

print('--'*30)
for element in reverse_idxs:
    print(element)
print('--'*30)
for bang in range(100):
    ostatok = bang % 4
    if bang % 4 == 0 and bang % 3 == 0:
        print("BANG! Hi-Hat!")
    elif bang % 4 == 0:
        print("BANG!")
    elif bang % 3 == 0:
        print("Hi-Hat!")
    else:
        print("---")

print ("Привет мир")