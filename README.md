# Landscape Categorization Model

A classification algorithm in supervised machine learning that utilizes pattern recognition techniques and statistical methods.

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Table of contents
* [Landscape Categorization Model](#landscape-categorization-model)
  * [Table of contents](#table-of-contents)
  * [Project Description](#project-description)
    * [Constraints](#constraints)
  * [Algorithm explanation](#algorithm-explanation)
    * [Features chosen](#features-chosen)
    * [Pattern recognition and model calculation algorithm](#pattern-recognition-and-model-calculation-algorithm)
    * [Implementation](#implementation)
  * [Resulting model](#resulting-model)
  * [Testing](#testing)
    * [Results](#results)
  * [Conclusion](#conclusion)
  * [Author](#author)
  * [License](#license)

## Project description

This project centers on creating, implementing, and testing a supervised machine learning algorithm designed to identify urban-natural landscapes. Utilizing pattern recognition and statistical methods, the algorithm aims to accurately classify images based on their color profiles. By harnessing machine learning, this approach has the potential to automate landscape identification, allowing for efficient analysis of large datasets. The goal is for the trained model to effectively determine whether a given picture depicts an urban-natural landscape.

### Constraints

For this project the following set of constraints were imposed:
  * The model may only have up to 3 features.
  * Selected features must rely solely in color and light, all edge-based features are out the scope.
  * Training dataset must be limited and high resolution, up to 10 images.

## Algorithm explanation

### Features chosen

### Pattern recognition and model calculation algorithm
  
  
### Implementation
The project was implemented using python v3.12.2 and the two main libraries used were OpenCV for image processing and Numpy for matrix operations. 

## Resulting model

  
These will be the test subject in the next section.

## Testing

The objetive of testing in this project is to validate the accuracy of the generated model. For this, a small dataset is used comprised of 10 high resolution images, which contain 50% landscapes and 50% non-landscape images such as animals or random settings. 

### Results

Using the automated testing process within the recognition module and the dataset detailed in the above section  the following results were found: 

| **Truth Value** | *Absolute Freq.* |
|-----------------|:----------------:|
| True positive   |        3         |
| True negative   |        4         |
| False positive  |        1         |
| False negative  |        2         |

According to the above table, the accuracy of the model is approximately 70%. 



Landscape Detector
===================

The current project aims to detect urban-rural landscape images using pattern recognition based in supervised machine learning with a Gaussian Naive Bayes (GNB) classifier. This will be achieved by implementing a data preprocessor, a feature extractor and a recognizer in Python. The resulting recognition model will then be tested against a new dataset to determine its accuracy level.

[//]: <> (Badges should go here)

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![code size](https://img.shields.io/github/languages/code-size/joshgc19/landscape_recognition_model)

# Table of contents
* [Landscape Detector](#landscape-detector)
  * [Table of contents](#table-of-contents)
  * [Installation](#installation)
  * [Data](#data)
    * [Source Data](#source-data)
    * [Data Preprocessing](#data-preprocessing)
  * [Code Structure](#code-structure)
  * [Implementation Overview](#implementation-overview)
    * [Design Constraints](#design-constraints)
    * [Chosen Features](#chosen-features)
  * [Results](#results)
    * [Training](#training)
      * [Model](#model)
    * [Testing](#testing)
  * [Conclusion](#conclusion)
  * [Project Author](#project-author)
  * [License](#license)

# Installation


# Data
To fully understand the features chosen in the current project, the following features of the type of landscapes that the model will be able to recognize have to be reviewed:
  
  * The picture must have the following hierarchical structure from top to bottom:
    * Blue daylight sky at the top with few clouds.
    * Solid ground must be portrayed with dominance of nature at the middle.
    * Buildings and housing may be present at the middle-bottom level.
  * Bodies of water may not be visible.
  * Depicted nature must be alive or green.
  * No snow coverage must be shown.
## Source Data
Make a list of all data sources with links and description
## Data Acquisition
If the acquisition of data is made through scraping or multiple API calls here is where this process will be stated. 
## Data Preprocessing
All operations made to the data before using

# Code Structure

```bash
├── data
│   ├── testing 
│   └── training
├── model_vectors
│   ├── Individuales.txt
│   └── Modelo_reconocedor_de_paisajes
├── preprocessed_data
│   ├── testing 
│   └── training
├── features_extraction.py
├── main.py
├── preprocessing.py
├── recognition.py
├── utils.py
├── LICENSE
├── README.md
└── .gitignore
```

# Implementation Overview
## Design Constraints
For this project the following design constraints were set aiming to create a recognition model with limited information:
  * The model may only have up to 3 features.
  * Selected features must rely solely in color and light, all edge-based features are out the scope.
  * Training dataset must be limited and high resolution, up to 10 images.
  * Accuracy level obtained must be greater than 50%.

## Chosen Features
The features chosen to be part of the features vector used throughout the project will be the following:
  * **Sky percentage (SP)**: This feature represents the percentage of the image in which the sky is depicted. For this purpose a 50 stripes gradient must be calculated from top to bottom in search of a delta greater than 65 in _RGB_ scale.
  * **Greeness (G)**: Percentage of green pixels contained in the image, for this project the color green will have the following _RGB_ color bounds: $G_L = (25, 25, 20)$ y $G_U =  (230, 255, 86)$.
  * **Skyness (S)**: Percentage of blueish pixels contained in the image, for this project the color sky or blueish will have the following _RGB_ color bounds: $S_L = (170, 45, 100)$ y $S_U =  (255, 255, 255)$.

And according to the above defined features, the feature vector that will be used is the following:

$$\vec{F} = (SP, G, S)$$

This features were chosen accounting for the design constraints present in the *[Design Constraints](#design-constraints)* section and the characterization of the dataset presente in the *[Data](#data)* section.

# Results
## Training
### Model
As a result of the training stage and each individual features vector, the average vector ( $E(\vec{F})$ ) which reflect central tendencies of the datas set and the variance vector ($\sigma_{\Vec{F}}$) which represents the variability across the data set. Both resulting vector are shown down below:
    
$$E(\vec{F}) = (0.32, 0.3315, 0.2601)$$

$$\sigma_{\vec{F}} = (0.1841, 0.0540, 0.1428)$$
## Testing

# Conclusion
As a summary, the preprocessing stage is critically important as it significantly impacts the quality of feature extraction, which forms the foundation of the model. Ensuring the correct selection of features and processing of data leads to a more precise model. Conversely, if the preprocessing stage is not carefully executed, the model may end up with a suboptimal set of features, leading to poor classification performance. Therefore, it is crucial to pay close attention to the preprocessing stage and features selection.

## Project author

**Joshua Gamboa Calvo**<br>
BS in Computing Engineering undergraduate<br>
Instituto Tecnológico de Costa Rica<br>
[LinkedIn](https://www.linkedin.com/in/joshgc19) | [GitHub](https://github.com/joshgc.19)<br>


## License
>You can checkout the full license [here (opens in the same tab)](https://github.com/joshgc19/landscape_recognition_model/blob/master/LICENSE). 

This project is licensed under the terms of the **MIT** license. 