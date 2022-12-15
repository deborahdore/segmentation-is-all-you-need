import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from skimage.segmentation import watershed
from sklearn.cluster import estimate_bandwidth, MeanShift

from src.segmentation.utils.img_utils import add_coordinates

warnings.filterwarnings("ignore")

WATERSHED = "/watershed_segmentation.svg"
THRESHOLDING = "/thresholding_segmentation.svg"
CLUSTERING = "/clustering.svg"
CLUSTERING_WITH_COORDINATES = "/clustering_coords.svg"
MEAN_SHIFT_SEGMENTATION = "/mean_shift_segmentation.svg"


def thresholding(imgs_preprocessed, plot_dir):
    """
    > The function applies a thresholding method to the images

    :param imgs_preprocessed: the images after preprocessing
    :param plot_dir: the directory where the plots will be saved
    """

    print("APPLYING METHOD THRESHOLDING")

    fig, ax = plt.subplots(5, 3, figsize=(16, 9), sharex=True, sharey=True)

    for i in range(len(imgs_preprocessed)):
        # white balancing
        percentile = 99
        white_patch_image = img_as_ubyte(
            (imgs_preprocessed[i] * 1.0 / np.percentile(imgs_preprocessed[i], percentile, axis=(0, 1))).clip(0, 1))

        # to gray scale
        gray = cv2.cvtColor(white_patch_image, cv2.COLOR_BGR2GRAY)

        # blur for better results
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # invert mask
        th3 = cv2.bitwise_not(th3)

        # kernel for dilatation
        kernel = np.ones((5, 5), np.uint8)
        # dilatate
        th3 = cv2.dilate(th3, kernel, iterations=2)

        segmented = cv2.bitwise_and(imgs_preprocessed[i], imgs_preprocessed[i], mask=th3)

        # save results
        ax[i][0].imshow(imgs_preprocessed[i].astype('uint8'))
        ax[i][0].set_xlabel('original image')
        ax[i][0].axis('off')

        ax[i][1].imshow(th3, cmap='gray')
        ax[i][1].set_xlabel('mask')
        ax[i][1].axis('off')

        ax[i][2].imshow(segmented)
        ax[i][2].set_xlabel('segmented')
        ax[i][2].axis('off')

        if i == len(imgs_preprocessed) - 1:
            # plt.show()
            plt.savefig(plot_dir + THRESHOLDING, dpi=1200)
            plt.close(fig)


def watershed_segmentation(img_preprocessed, plot_dir):
    """
    > The watershed algorithm is a classic algorithm used for segmentation a

    :param img_preprocessed: the preprocessed image
    :param plot_dir: the directory where the plots will be saved
    """

    print("APPLYING METHOD WATERSHED")

    fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(16, 9), sharex=True, sharey=True)

    for index in range(len(img_preprocessed)):

        # to gray scale
        gray = cv2.cvtColor(img_preprocessed[index], cv2.COLOR_RGB2GRAY)

        # denoise image
        denoised = rank.median(gray, disk(2))

        # create markers
        markers = rank.gradient(denoised, disk(5)) < 15
        markers = ndimage.label(markers)[0]

        # follow gradient
        gradient = rank.gradient(denoised, disk(5))

        # apply watershed
        labels = watershed(gradient, markers)

        # save image
        ax[index][0].imshow(gray, cmap=plt.cm.gray)
        ax[index][0].set_title("Original")
        ax[index][0].axis('off')

        ax[index][1].imshow(gradient, cmap=plt.cm.nipy_spectral)
        ax[index][1].set_title("Local Gradient")
        ax[index][1].axis('off')

        ax[index][2].imshow(markers, cmap=plt.cm.nipy_spectral)
        ax[index][2].set_title("Markers")
        ax[index][2].axis('off')

        ax[index][3].imshow(gray, cmap=plt.cm.gray)
        ax[index][3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
        ax[index][3].set_title("Segmented")
        ax[index][3].axis('off')

        fig.tight_layout()

        if index == len(img_preprocessed) - 1:
            # plt.show()
            plt.savefig(plot_dir + WATERSHED, dpi=1200)
            plt.close(fig)


def clustering(image, plot_dir):
    """
    It takes an image and returns a list of images, each of which is the result of applying k-means clustering to the
    original image with a different number of clusters

    :param image: The image to be segmented
    :param plot_dir: The directory where the plots will be saved
    """
    print("APPLYING METHOD CLUSTERING")
    h = image.shape[0]
    w = image.shape[1]
    channels = image.shape[2]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    result_image = []
    for K in range(1, 17):
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image.append(res.reshape(h, w, channels))

    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    ax = axes.ravel()

    for index, img in enumerate(result_image):
        ax[index].imshow(img)
        ax[index].set_title(f'Cluster {index + 1}')
        ax[index].axis('off')

    fig.tight_layout()
    plt.savefig(plot_dir + CLUSTERING, dpi=1200)
    plt.close(fig)

    plt.imshow(result_image[4])
    plt.axis('off')
    plt.title("Cluster 5")
    plt.savefig(plot_dir + "/clustering_5.svg", dpi=1200)


def clustering_coords(image, plot_dir):
    """
    It takes an image, adds coordinates to it, and then applies k-means clustering to it

    :param image: the image to be segmented
    :param plot_dir: The directory where the plots will be saved
    """
    print("APPLYING METHOD CLUSTERING")
    h = image.shape[0]
    w = image.shape[1]
    channels = 5  # adding coords

    percentile = 99
    image = img_as_ubyte((image * 1.0 / np.percentile(image, percentile, axis=(0, 1))).clip(0, 1))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = add_coordinates(img)
    vectorized = img.reshape((-1, channels))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    result_image = []
    for K in range(1, 17):
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        reshape = res.reshape(h, w, channels)
        result_image.append(reshape[:, :, 0:3])

    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    ax = axes.ravel()

    for index, img in enumerate(result_image):
        ax[index].imshow(img)
        ax[index].set_title(f'Cluster_{index + 1}')
        ax[index].axis('off')

    fig.tight_layout()
    plt.savefig(plot_dir + CLUSTERING_WITH_COORDINATES, dpi=1200)
    plt.close(fig)


def mean_shift_clustering(img_preprocessed, plot_dir):
    """
    It takes an image and returns a new image to which mean shift clustering was applied

    :param img_preprocessed: The image that we want to segment
    :param plot_dir: The directory where the plots will be saved
    """
    print("APPLYING METHOD MEAN SHIFT CLUSTERING")
    h = img_preprocessed.shape[0]
    w = img_preprocessed.shape[1]
    c = img_preprocessed.shape[2]

    img_preprocessed = img_preprocessed.reshape(-1, c)

    bandwidth = estimate_bandwidth(img_preprocessed, quantile=0.3, n_samples=300, n_jobs=-1, random_state=32)

    print(f'Bandwidth: {bandwidth}')

    msc = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1).fit(img_preprocessed)

    centers = msc.cluster_centers_[msc.labels_]
    n_clusters = len(msc.cluster_centers_)

    final_image = centers.reshape(h, w, c)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_preprocessed.reshape(h, w, c))
    ax[0].set_xlabel('Original')
    ax[0].axis('off')

    ax[1].imshow(final_image.astype('uint8'))
    ax[1].set_xlabel(f'MeanShift with {n_clusters} clusters')
    ax[1].axis('off')
    # plt.show()

    plt.savefig(plot_dir + MEAN_SHIFT_SEGMENTATION, dpi=1200)
    plt.close(fig)
