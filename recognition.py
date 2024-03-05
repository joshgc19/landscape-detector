import numpy as np

from preprocessing import preprocess
from features_extraction import extract_features
from utils import VECTORS_DIR, MODEL_FILE_NAME, read_model_vectors


def parse_vector(unparsed_vector):
    return np.array(list(map(float, unparsed_vector.replace("\n", "").split(","))))


def load_model_vectors():
    unparsed_model_vectors = read_model_vectors(VECTORS_DIR + MODEL_FILE_NAME)
    average_vector = parse_vector(unparsed_model_vectors[0])
    variance_vector = parse_vector(unparsed_model_vectors[1])
    return average_vector - variance_vector, average_vector + variance_vector


def recognize():
    preprocess(True)
    features_list = extract_features(True)
    lower_bound_vector, upper_bound_vector = load_model_vectors()
    target = [True, True, True, True, True, False, False, False, False, False]

    idx = 0
    false_positives = 0
    false_negatives = 0
    truth_positives = 0
    truth_negatives = 0

    print("== Recognition results ==\n")
    for image_name, features in features_list:
        print("Image name - Feature: ", image_name, " - ", features)

        is_lower_bound_compliant = lower_bound_vector <= features
        is_upper_bound_compliant = upper_bound_vector >= features
        is_landscape = is_lower_bound_compliant.all() and is_upper_bound_compliant.all()

        print("Is landscape? Test result:", is_landscape.all(), " Target value: ", target[idx])

        if is_landscape.all() and target[idx]:
            truth_positives += 1
        elif not is_landscape.all() and not target[idx]:
            truth_negatives += 1
        elif is_landscape.all() and not target[idx]:
            false_positives += 1
        elif not is_landscape.all() and target[idx]:
            false_negatives += 1
        idx += 1

    print("\n== Statistics ==\n")
    print("Truth positives = ", truth_positives, " observations - ", truth_positives/idx*100, "%")
    print("Truth negatives = ", truth_negatives, " observations - ", truth_negatives/idx*100, "%")
    print("False positives = ", false_positives, " observations - ", false_positives / idx * 100, "%")
    print("False negatives = ", false_negatives, " observations - ", false_negatives / idx * 100, "%")
    print("Accuracy = ",  (truth_positives + truth_negatives) / idx * 100, "%")


recognize()