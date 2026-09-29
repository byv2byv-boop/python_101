with open("data/onegin.txt", encoding="utf-8") as file:
    onegin = file.read()

VOWELS = "аеёиоуыэюя"

letters = ""
for char in onegin.lower():
    if "а" <= char <= "я" or char == "ё":
        letters += char

letters = letters[:20000]
print(f"Количество взятых букв: {len(letters)}")

#На глассные и согласные делим: "Г" или "С"
kinds = ""
for char in letters:
    if char in VOWELS:
        kinds = kinds + "Г"
    else:
        kinds += "С"

print(f"НАЧАЛО СТРОКИ: {kinds[:40]}")

transitions = {"ГГ": 0, "ГС": 0, "СГ": 0, "СС": 0}
for i in range(len(kinds) - 1):
    pair = kinds[i] + kinds[i+1]
    transitions[pair] += 1

print("Переходы")
for t in transitions:
    print(transitions[t])
for pair in transitions:
    print(pair, transitions[pair])


print("Hello world!")