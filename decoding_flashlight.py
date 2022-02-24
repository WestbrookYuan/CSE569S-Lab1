import time

import cv2
import os
import ckwrap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat


def video_to_images(input_video_dir, output_folder_dir):
    num = 1
    vid = cv2.VideoCapture(input_video_dir)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            file_name = '%08d' % num
            cv2.imwrite(output_folder_dir + "\\" + str(file_name) + '.jpg', frame)
            num += 1
        else:
            break


def brightness(im_file):
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def plot_brightness(x, y):
    plt.figure(figsize=(15, 5))
    plt.title('Brightness per Frame', fontsize=20)
    plt.xlabel(u'frame', fontsize=14)
    plt.ylabel(u'brightness', fontsize=14)
    plt.plot(x, y)
    plt.show()


def brightness_to_lengths(threshold, brightness_per_frame):
    groups = []
    symbols = []

    tempTrue = []
    tempFalse = []
    for i in brightness_per_frame:
        if i >= threshold:
            tempTrue.append(i)
            if tempFalse:
                groups.append(tempFalse)
                tempFalse = []
        else:
            tempFalse.append(i)
            if tempTrue:
                groups.append(tempTrue)
                tempTrue = []
    if brightness_per_frame[-1] < threshold:
        groups.append(tempFalse)
    else:
        groups.append(tempTrue)
    for i in groups:
        if i[0] >= threshold:
            symbols.append(len(i))
        else:
            symbols.append(len(i) * -1)
    return symbols


def classify_symbols(symbols, k=5, method='linear'):
    return ckwrap.ckmeans(symbols, k=k, method=method).labels


morse_to_letter = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G',
                   '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N',
                   '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
                   '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z',
                   '.-.-.-': '.', '--..--': ',', '-.-.--': '!', '..--..': '?', '-..-.': '/', '.--.-.': '@',
                   '.----.': '\'',
                   '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5',
                   '-....': '6', '--...': '7', '---..': '8', '----.': '9', '-----': '0'}

label_to_morse = {0: " " * 7, 1: " " * 3, 2: " ", 3: ".", 4: "-"}


def morse_to_plaintext(morse):
    words = "".join(morse).split("0")
    letters = []
    for i in words:
        if i:
            letters.append(i.split("1"))
        else:
            letters.append(i)
    sentence = []
    for i in range(len(letters)):
        if letters[i] == '':
            sentence.append(" ")
        else:
            temp = []
            for j in letters[i]:
                for k in j:
                    if k != "2":
                        temp.append(k)
                sentence.append("".join(temp))
                temp = []
            sentence.append(" ")

    replaced_sentence = []
    for i in range(len(sentence)):
        temp = []
        if sentence[i] != " ":
            for j in sentence[i]:
                for k in j:
                    if int(k) in label_to_morse:
                        temp.append(label_to_morse[int(k)])
            replaced_sentence.append("".join(temp))
        elif sentence[i] == " ":
            replaced_sentence.append(sentence[i])

    sentence = []
    for i in replaced_sentence:
        if i in morse_to_letter:
            sentence.append(morse_to_letter[i])
        else:
            sentence.append(i)
    return "".join(sentence)


def run(input_dir):
    t = time.time()
    video_to_images(input_dir, './outputs')

    path = "./outputs"

    image_list = os.listdir(path)
    brights = []
    for i in range(len(image_list)):
        brights.append(brightness(path + "/" + image_list[i]))
    print(" Checkpoint 1: brightness plot")
    plot_brightness([i for i in range(len(brights))], brights)

    symbols = brightness_to_lengths(150, brights)
    print("Checkpoint 2: a labeled list of signal lengths is ", symbols)

    morse = classify_symbols(symbols, k=5)
    print(morse)
    print(morse_to_plaintext(list(map(str, morse.tolist()))))
    print(time.time() - t)


run("./inputs/encoded.mov")
