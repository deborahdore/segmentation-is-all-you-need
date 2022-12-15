from src.segmentation.config.config_dir import ConfigDirectory
from src.segmentation.segmentation import thresholding, watershed_segmentation, clustering, clustering_coords, \
    mean_shift_clustering
from src.segmentation.utils.img_utils import img_preprocessing
from src.segmentation.utils.utils import load_random_images, print_time


def main():
    """
    It loads a random image from the dataset, preprocesses it, and then runs the thresholding, watershed segmentation,
    clustering, and mean shift clustering algorithms on it
    """
    directory = ConfigDirectory()
    images = load_random_images(directory.get_dataset_dir())
    imgs_processed = []

    for image in images:
        imgs_processed.append(img_preprocessing(image))

    plot_dir = directory.get_plot_dir()

    print_time("START")
    thresholding(imgs_processed, plot_dir)

    print_time("THRESHOLDING", directory.get_report_dir())

    watershed_segmentation(imgs_processed, plot_dir)

    print_time("WATERSHED", directory.get_report_dir())

    clustering(imgs_processed[0], plot_dir)

    print_time("CLUSTERING", directory.get_report_dir())

    clustering_coords(imgs_processed[0], plot_dir)

    print_time("CLUSTERING WITH COORDINATES", directory.get_report_dir())

    mean_shift_clustering(imgs_processed[0], plot_dir)

    print_time("MEAN SHIFT CLUSTERING", directory.get_report_dir())

    print_time("END")


if __name__ == '__main__':
    main()
