import cv2

from preprocessing import preprocess
from features_extraction import extract_features
from recognition import recognize

if __name__ == '__main__':
    preprocess()
    extract_features()
