import cv2
import numpy as np

from src.segmentation.utils.utils import load_image_folder


def img_preprocessing(image):
    """
    It takes an image, crops it, blurs it, resizes it, and normalizes it

    :param image: The image to be processed
    :return: The image is being cropped, blurred, resized, and normalized.
    """

    width, height = image.size

    left = (width - width / 1.2) / 2
    top = (height - height / 1.2) / 2
    right = (width + width / 1.2) / 2
    bottom = (height + height / 1.2) / 2

    # Crop the center of the image
    im = image.crop((left, top, right, bottom))

    im = np.array(im)

    #  16/9
    blur = cv2.GaussianBlur(im, (3, 3), 0)

    dsize = (240, 135)
    resize = cv2.resize(blur, dsize, interpolation=cv2.INTER_AREA)

    norm = np.zeros(dsize)
    norm_image = cv2.normalize(resize, norm, 0, 255, cv2.NORM_MINMAX)

    return norm_image


def add_coordinates(img_processed):
    """
    It takes an image and adds the coordinates of each pixel to the end of the pixel's feature vector

    :param img_processed: the image that has been processed by the preprocessing function
    :return: a new image with the coordinates of each pixel appended to the end of the pixel's array.
    """
    img_new = np.empty((img_processed.shape[0], img_processed.shape[1], 5))
    for k in range(0, img_processed.shape[0]):
        for j in range(0, img_processed.shape[1]):
            img_new[k, j] = np.append(img_processed[k, j], [k, j])
    return img_new


def find_mean_std_df(dataset_dir):
    """
    It takes a directory of images, and returns the mean and standard deviation of the pixel values of those images

    :param dataset_dir: The directory where the dataset is stored
    :return: The mean and standard deviation of the dataset.
    """
    files = load_image_folder(dataset_dir)

    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])

    numSamples = len(files)

    for im in files:
        img = np.array(im).astype(float) / 255

        for j in range(3):
            mean[j] += np.mean(img[:, :, j])

    mean = (mean / numSamples)
    print(f"Mean: {mean}")

    for im in files:
        img = np.array(im).astype(float) / 255

        for j in range(3):
            stdTemp[j] += ((img[:, :, j] - mean[j]) ** 2).sum() / (img.shape[0] * img.shape[1])

    std = np.sqrt(stdTemp / numSamples)

    print(f"STD: {std}")

    return mean, std
