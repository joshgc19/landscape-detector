import os
import math
import numpy as np

# DATA AND IMAGES FOLDERS PATH
TRAINING_DATA_DIR = "./data/training/"
TESTING_DATA_DIR = "./data/testing/"

PREPROCESS_TRAINING_DATA_DIR = "./preprocessed_data/training/"
PREPROCESS_TESTING_DATA_DIR = "./preprocessed_data/testing/"

VECTORS_DIR = "./model_vectors/"

INDIVIDUAL_VECTOR_FILE_NAME = "individuals"
MODEL_FILE_NAME = "obtained_model"


def sum_tuples(array):
    """
    Function that sums a list of tuples by their respective indexes
    :param array: List of tuples
    :return: A single tuple containing the result tuple
    """
    if len(array) == 0:
        return None
    grouped_tuple_value_by_idx = [[tup[i] for tup in array] for i in range(len(array[0]))]
    return tuple(sum(i) for i in grouped_tuple_value_by_idx)


def tuple_division(tup, divisor):
    """
    Function that divides a tuple by a given number, does not throw an exception for division by zero
    :param tup: Tuple to be divided
    :param divisor: Number used as division factor
    :return: reduced tuple
    """
    if divisor == 0:
        return tup
    return tuple(x / divisor for x in tup)


def retrieve_filenames_from_directory(directory):
    """
    Function that returns a list of files from a given directory
    :param directory: target directory
    :return: List of strings containing the names of the files in the directory
    """
    return os.listdir(directory)


def write_into_file(filename, data):
    """
    Procedure that writes data into a given file
    :param filename: name of the file
    :param data: data to be written to the file
    """
    with open(filename + ".txt", "w") as file:
        file.write("\n".join(data))


def read_model_vectors(filename):
    """
    Function that reads the lines of a file and returns them
    :param filename: name of the file to be accessed
    :return: list of strings with the contents of the file
    """
    with open(filename + ".txt", "r") as file:
        return file.readlines()


def join_list_floats(floats, separator):
    """
    Function thar joins a list of floats as a string
    :param floats: list of floats to be joined into string
    :param separator: string used to separate each float
    :return: string with joined floats
    """
    return separator.join([str(flt) for flt in floats])

