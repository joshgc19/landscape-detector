from preprocessing import preprocess
from features_extraction import extract_features
from recognition import recognize

if __name__ == '__main__':
    # Training data and model training
    preprocess()
    extract_features()
    # Testing process with testing data subset and the retrieved model
    recognize()
