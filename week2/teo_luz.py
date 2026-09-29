#!/usr/bin/python

# Stochastic Texts
# copyright (c) 2014, 2024 Nick Montfort <nickm@nickm.com>
# inspired by a 1959 program by Theo Lutz; text translated by Helen MacCormack
# intended to work in Python 2 (>= 2.5) as well as 3, however:
#
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved. This file is offered as-is,
# without any warranty.
#
# Updated 31 May 2018 for compatibility with Python 2 (2.5+) and 3.
# Updated 1 January 2024 for Memory Slam 2.0.

from random import choice
from sys import argv
from time import sleep

subjects = ["count", "stranger", "look", "church", "castle", "picture",
            "eye", "village", "tower", "farmer", "way", "guest", "day",
            "house", "table", "labourer"]
predicates = ["open", "silent", "strong", "good", "narrow", "near",
              "new", "quiet", "far", "deep", "late", "dark", "free",
              "large", "old", "angry"]
conjunctions = [" and ", " or ", " therefore ", ". ", ". ", ". ", ". ", ". "]
operators = ["a", "every", "no", "not every"]

def phrase():
    text = choice(operators) + " " + choice(subjects)
    if text == "a eye":
        text = "an eye"
    return text + " is "

def element():
    text = phrase() + choice(predicates)
    return text[0].upper() + text[1:]

def line():
    first = element()
    conjunction = choice(conjunctions)
    second = element()
    if not conjunction == ". ":
        second = second.lower()
    text = first + conjunction + second + ".\n"
    return text.upper() if (len(argv) > 1 and argv[1] == '-c') else text

print()
while True:
    print(line())
    sleep(3.0)