with open("data/onegin.txt", "r", encoding="utf-8") as file:
    onegin = file.read()

print(type(onegin))
print(f"Количество симвлов: {len(onegin)}")

lines = onegin.split('\n')
print(f"Количество строк: {len(lines)}")

words = onegin.split()
print(f"Количество слов: {len(words)}")

print(f"Первые 300 симвловов")
print(onegin [:300])

print("hellow world")