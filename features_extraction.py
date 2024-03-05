import math
import numpy as np
import cv2

from utils import PREPROCESS_TESTING_DATA_DIR, PREPROCESS_TRAINING_DATA_DIR, VECTORS_DIR, INDIVIDUAL_VECTOR_FILE_NAME, MODEL_FILE_NAME, retrieve_filenames_from_directory, write_into_file, sum_tuples, tuple_division, join_list_floats

COLOR_MIN_DELTA = 60


def read_image_extract_feature(feature_function, image_path):
    image = cv2.imread(image_path)
    return feature_function(image)


def calculate_average_feature_vector(feature_list):
    sum_value = sum_tuples(feature_list)
    average_vector = tuple_division(sum_value, len(feature_list))
    return average_vector


def calculate_variance_feature_vector(feature_list, average_vector):
    variance_vector = []

    feature_count = len(average_vector)
    observations_count = len(feature_list)

    for i in range(feature_count):
        differences = np.array([feature[i] - average_vector[i] for feature in feature_list])
        variance_vector.append(math.sqrt((sum(differences**2)/observations_count)))

    return variance_vector


def extract_skyline(image):
    image = image.reshape((-1, 3))
    pixel_count = np.size(image) // 3
    for i in range(pixel_count - 1):
        delta = abs(sum(image[i]) - sum(image[i + 1]))
        if delta >= COLOR_MIN_DELTA:
            return (i + 1) / pixel_count
        elif i >= 1 and abs(sum(image[i - 1]) - sum(image[i + 1])) >= COLOR_MIN_DELTA:
            return (i + 1) / pixel_count
    return 0


def extract_color_percentage(image):
    return np.count_nonzero(image) / np.size(image)


def extract_features(is_testing=False):
    data_dir = PREPROCESS_TESTING_DATA_DIR if is_testing else PREPROCESS_TRAINING_DATA_DIR
    filenames = retrieve_filenames_from_directory(data_dir)

    images_feature_files = {}

    for i in range(0, len(filenames), 3):
        images_feature_files[filenames[i].split("_")[0]] = filenames[i:i + 3]

    feature_values = []
    images_names = []
    function_list = [extract_skyline, extract_color_percentage, extract_color_percentage]

    for image_name, image_dimensions in images_feature_files.items():
        images = list(map(lambda x: cv2.imread(data_dir + x), image_dimensions))
        features = list(map(lambda func, arg: func(arg), function_list, images))
        feature_values.append(features)
        images_names.append(image_name)

    if not is_testing:
        average_vector = calculate_average_feature_vector(feature_values)
        variance_vector = calculate_variance_feature_vector(feature_values, average_vector)

        write_into_file(VECTORS_DIR + INDIVIDUAL_VECTOR_FILE_NAME, list(map(lambda feature_list: join_list_floats(feature_list,","), feature_values)))
        write_into_file(VECTORS_DIR + MODEL_FILE_NAME, [join_list_floats(average_vector, ","), join_list_floats(variance_vector, ",")])
    else:
        return zip(images_names, feature_values)


