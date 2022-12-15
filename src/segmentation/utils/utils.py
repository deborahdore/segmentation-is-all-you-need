import glob
import os
import random
import time
from datetime import date
from os.path import join

from PIL import Image

CLASSES = ["alfa_romeo", "bwt", "ferrari", "haas", "mclaren",
           "mercedes", "redbull", "renault", "toro_rosso", "williams"]

TIME = time.time()


def print_time(additional_info="", report_dir=None):
    """
    It prints the time elapsed since the last time it was called

    :param additional_info: a string that will be printed along with the execution time
    :param report_dir: the directory where the report will be saved
    """
    global TIME
    exec_time = f'{additional_info} execution time: {time.time() - TIME} seconds'
    print(exec_time)
    if report_dir is not None:
        with open(report_dir + f"/{date.today()}", "a") as file:
            file.write(exec_time)
            file.write("\n")
    TIME = time.time()


def load_image(path):
    """
    It loads an image from a file and returns it

    :param path: The path to the image you want to load
    :return: The image is being returned.
    """
    return Image.open(path)


def load_random_images(dataset_path):
    """
    It loads 5 random images from the dataset

    :param dataset_path: The path to the dataset
    :return: A list of 5 random images from the dataset.
    """
    teams = random.sample(CLASSES, k=5)
    images = []
    for team in teams:
        folder_path = join(dataset_path, team)
        image = join(folder_path, random.choice(os.listdir(folder_path)))
        images.append(Image.open(image))
    return images


def load_image_folder(folder):
    """
    It takes a folder name as input, and returns a list of all the images in that folder

    :param folder: the folder that contains the images
    :return: A list of images
    """
    image_list = []
    for filename in glob.glob(folder + "/*/" + '*.jpg'):
        im = Image.open(filename)
        image_list.append(im)
    return image_list
