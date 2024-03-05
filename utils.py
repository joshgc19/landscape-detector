import os

TRAINING_DATA_DIR = "./data/training/"
TESTING_DATA_DIR = "./data/testing/"

PREPROCESS_TRAINING_DATA_DIR = "./preprocessed_data/training/"
PREPROCESS_TESTING_DATA_DIR = "./preprocessed_data/testing/"

VECTORS_DIR = "./model_vectors/"

INDIVIDUAL_VECTOR_FILE_NAME = "Individuales"
MODEL_FILE_NAME = "Modelo_reconocedor_de_paisajes"


def sum_tuples(array):
    if len(array) == 0:
        return None
    grouped_tuple_value_by_idx = [[tup[i] for tup in array] for i in range(len(array[0]))]
    return tuple(sum(i) for i in grouped_tuple_value_by_idx)


def tuple_division(tup, divisor):
    if divisor == 0:
        return tup
    return tuple(x / divisor for x in tup)


def retrieve_filenames_from_directory(directory):
    return os.listdir(directory)


def write_into_file(filename, data):
    with open(filename + ".txt", "w") as file:
        file.write("\n".join(data))


def read_model_vectors(filename):
    with open(filename + ".txt", "r") as file:
        return file.readlines()


def join_list_floats(floats, separator):
    return separator.join([str(flt) for flt in floats])