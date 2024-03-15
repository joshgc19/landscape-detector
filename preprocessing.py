import numpy as np
import cv2

from utils import TRAINING_DATA_DIR, TESTING_DATA_DIR, PREPROCESS_TRAINING_DATA_DIR, PREPROCESS_TESTING_DATA_DIR, \
    sum_tuples, tuple_division, retrieve_filenames_from_directory


def isolate_color_mask(image, lower_bound, upper_bound, image_name, color_name):
    """
    Function used to isolate a given RGB color range from an image
    :param image: numpy matrix containing the color values for each pixel in the image
    :param lower_bound: tuple containing the lower bound of the color to isolate
    :param upper_bound: tuple containing the upper bound of the color to isolate
    :param image_name: name of the source input image
    :param color_name: name of the color to distinguish between color masks
    :return: a mask of the isolated color
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Mask of desired colors bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_pixel_count = np.count_nonzero(mask) / 3
    # Slice the mask out of the image
    imask = mask > 0
    # Create empty image
    color_masked = np.zeros_like(image, np.uint8)
    # Add masked pixels to empty image
    color_masked[imask] = image[imask]
    # Save image
    cv2.imwrite(image_name + "_" + color_name + ".jpg", color_masked)
    return mask_pixel_count


def compute_average_stripes(image, stripe_count, image_name):
    """
    Procedure that collapses an image into a vertical gradient of average colors
    :param image: numpy matrix containing the color values for each pixel in the image
    :param stripe_count: It represents in how many blocks the image will be partitioned
    :param image_name: name of the source input image
    """
    width, height, depth = image.shape
    # Calculate the strip size of an image with a given stripe count
    stripe_size = width // stripe_count
    # Create empty gradient image
    empty_image = np.zeros((stripe_count, 1, 3))
    # Loop to retrieve average stripe color
    for i in range(stripe_count):
        # Retrieve image sub matrix
        pixel_sublist = image[i * stripe_size:(i + 1) * stripe_size:, ::]
        # Flatten one dimension
        pixel_sublist = pixel_sublist.reshape((-1, 3))
        # Compute average RGB value for retrieved area
        average_pixel = tuple_division(sum_tuples(pixel_sublist), len(pixel_sublist))
        # Add pixel to gradient image
        if average_pixel is not None:
            empty_image[i] = average_pixel

    # Save result image
    cv2.imwrite(image_name + "_gradient.jpg", empty_image)


def preprocess(is_testing=False):
    """
    Procedure that preprocesses the images in the testing and training sets according to the preprocessing rules
    :param is_testing: Boolean to indicate which dataset has to be preprocessed
    """
    # Form directories paths needed
    data_dir = TESTING_DATA_DIR if is_testing else TRAINING_DATA_DIR
    target_dir = PREPROCESS_TESTING_DATA_DIR if is_testing else PREPROCESS_TRAINING_DATA_DIR
    # Retrieve file names and extension from directory
    files = retrieve_filenames_from_directory(data_dir)

    # Define color green upper and lower color bounds (BGR)
    green_lower_bound = (20, 25, 25)
    green_upper_bound = (86, 255, 230)

    # Define color sky upper and lower color bounds (BGR)
    sky_lower_bound = (100, 45, 170)
    sky_upper_bound = (255, 255, 255)

    # Loop used to preprocess all retrieved files
    for file in files:
        # Drop file extension to obtain filename
        file_name = file.split(".")[0]
        # Read image as a pixels matrix
        image = cv2.imread(data_dir + file)
        # Computing of the three variables preprocessed images
        compute_average_stripes(image, 50, target_dir + file_name)
        isolate_color_mask(image, green_lower_bound, green_upper_bound, target_dir + file_name, "green")
        isolate_color_mask(image, sky_lower_bound, sky_upper_bound, target_dir + file_name, "sky")
