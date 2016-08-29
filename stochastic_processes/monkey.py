import numpy as np
from numpy.random import binomial, randint

text = "This is a test"
alphabet = [l for l in "abcdefghijklmnopqrstuvwxyz "]
letters = [word for word in text]
letters
p = 0.1


index = 0
write = ""
i = 0
while True:
    if write == text:
        break
    if binomial(1, p) == 1:
        write += letters[index]
        index+=1
        if len(write) > 6:
            print(write, i, end = "\r")
    else:
        choose_letter = randint(low=0, high=len(alphabet))
        write += alphabet[choose_letter]
        index = 0
        #print(write)
        write = ""
    i += 1
