import numpy as np
from numpy.random import choice
from collections import OrderedDict
import pickle

alphabet = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".lower().split()
# The desired stream of words to be written
target_stream = "maths"

data_dict = OrderedDict()

# Iterate 50 instances of the monkey
for n_try in range(50):

    current_streak = ""
    streak_count = 0
    streaks_milestone = OrderedDict()
    count = 0
    max_streak = ""

    print("---------------------{}------------------------".format(n_try))
    while True:
        count += 1
        # Draw a single uniform sample from the alphabet
        sample_character = choice(alphabet)
        if sample_character == target_stream[streak_count]:
            streak_count += 1
            current_streak += sample_character

            # If a new streak is broken, save streak and iteration number
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

    # Save current data to data_dict
    data_dict["round_" + str(n_try)] = streaks_milestone

# Pickle it
with open("monkey.pickle", "wb") as f:
    pickle.dump(data_dict, f)
