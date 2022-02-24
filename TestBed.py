import numpy as np

from decoding_flashlight import video_to_images, brightness, plot_brightness, brightness_to_lengths, classify_symbols, \
    morse_to_plaintext
import os

# video_to_images('./inputs/encoded.mov', './outputs')  # one time

path = "./outputs"

image_list = os.listdir(path)
brights = []
for i in range(len(image_list)):
    brights.append(brightness(path + "/" + image_list[i]))

# plot_brightness([i for i in range(len(brights))], brights)

symbols = brightness_to_lengths(150, brights)
morse = classify_symbols(symbols, k=5)
print(morse_to_plaintext(list(map(str, morse.tolist()))))
