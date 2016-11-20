import numpy as np
from numpy.random import choice
from collections import OrderedDict

alphabet = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".lower().split()
# The desired stream of words to be written
target_stream = "wahrscheinlichkeit"
target_stream = "markov"

current_streak = ""
streak_count = 0

streaks_milestone = OrderedDict()

count = 0
max_streak = ""
while True:
    count += 1
    # Draw a single uniform sample from the alphabet
    sample_character = choice(alphabet)
    if sample_character == target_stream[streak_count]:
        streak_count += 1
        current_streak += sample_character

        # If a new streak is broken, save info
        if len(current_streak) > len(max_streak):
            streaks_milestone[current_streak] = count
            max_streak = current_streak

        # Print current state of simulation
        print(max_streak, count, end="\r")

    else:
        streak_count = 0
        current_streak = ""

    if len(current_streak) == len(target_stream):
        break


print(streaks_milestone)
