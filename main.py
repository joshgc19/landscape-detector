from preprocessing import preprocess
from features_extraction import extract_features
from recognition import recognize

# Main to train model
# if __name__ == '__main__':
#     # Training data and model training
#     preprocess()
#     extract_features()


# Main to test model
if __name__ == '__main__':
    # Retrival and calculation of feature vectors
    # preprocess(True)
    features_list = extract_features(True)
    # Testing process with testing data subset and the retrieved model
    recognize(features_list)