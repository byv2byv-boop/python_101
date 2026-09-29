morse_dict = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    " ": " ",

    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----."
}
our_text = "ALFA"

morse_text = [] #список в котором нет еще элементов
for letter in our_text:
    print(f"Наша бука: {letter}, Морзе:{morse_dict[letter]}")
    morse_text.append(morse_dict[letter])

    print(morse_text)

final_text = ' '.join(morse_text)
print(final_text)

def text_to_morse():
    our_input = input("Введите текст на английском: \n")
    our_input = our_input.upper()
    morse_text = []
    for letter in our_input:
        try:
            morse_letter = morse_dict[letter]
        except:
            print ("Введи нормальный текст на английском!")
            return

        morse_text.append(morse_letter)
    final_text = ' '.join(morse_text)
    print(f"Зашифрованный результат: {final_text}")
    return final_text