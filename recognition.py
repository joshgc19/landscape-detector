import numpy as np

from utils import VECTORS_DIR, MODEL_FILE_NAME, read_model_vectors


def parse_vector(unparsed_vector):
    """
    Function that parses the model vector stored in file as a ndarray
    :param unparsed_vector: a string containing the model vector separated by ","
    :return: ndarray containing the model vector
    """
    return np.array(list(map(float, unparsed_vector.replace("\n", "").split(","))))


def load_model_vectors():
    """
    Function that loads the model vectors stored in the file
    :return: lower and upper bounds of the model vector
    """
    # Retrieve model vector from file
    unparsed_model_vectors = read_model_vectors(VECTORS_DIR + MODEL_FILE_NAME)
    # Parse mean and variance vector
    average_vector = parse_vector(unparsed_model_vectors[0])
    variance_vector = parse_vector(unparsed_model_vectors[1])
    # return bounds of model vector
    return average_vector - variance_vector, average_vector + variance_vector


def recognize(features_list):
    """
    Function that classifies a set of images as a landscape or not landscape
    """
    # Loads the model vector
    lower_bound_vector, upper_bound_vector = load_model_vectors()
    # The target list
    target = [True, True, True, True, True, False, False, False, False, False]

    # Count variables
    idx = 0
    false_positives = 0
    false_negatives = 0
    truth_positives = 0
    truth_negatives = 0

    print("== Recognition results ==\n")
    # Loop that checks if the feature vector fits into the model vector and prints the result
    for image_name, features in features_list:
        print("Image name - Feature: ", image_name, " - ", features)

        is_lower_bound_compliant = lower_bound_vector <= features
        is_upper_bound_compliant = upper_bound_vector >= features
        is_landscape = is_lower_bound_compliant.all() and is_upper_bound_compliant.all()

        print("Is landscape? Test result:", is_landscape.all(), " Target value: ", target[idx])

        # Count false and truth negatives and positives
        if is_landscape.all() and target[idx]:
            truth_positives += 1
        elif not is_landscape.all() and not target[idx]:
            truth_negatives += 1
        elif is_landscape.all() and not target[idx]:
            false_positives += 1
        elif not is_landscape.all() and target[idx]:
            false_negatives += 1
        idx += 1

    # Prints out general statistics
    print("\n== Statistics ==\n")
    print("Truth positives = ", truth_positives, " observations - ", truth_positives / idx * 100, "%")
    print("Truth negatives = ", truth_negatives, " observations - ", truth_negatives / idx * 100, "%")
    print("False positives = ", false_positives, " observations - ", false_positives / idx * 100, "%")
    print("False negatives = ", false_negatives, " observations - ", false_negatives / idx * 100, "%")
    print("Accuracy = ", (truth_positives + truth_negatives) / idx * 100, "%")
