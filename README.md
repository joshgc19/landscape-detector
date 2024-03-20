Landscape Detector
===================

The current project aims to detect urban-rural landscape images using pattern recognition based in supervised machine learning with a Gaussian Naive Bayes (GNB) classifier. This will be achieved by implementing a data preprocessor, a feature extractor and a recognizer in Python. The resulting recognition model will then be tested against a new dataset to determine its accuracy level.

[//]: <> (Badges should go here)

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![code size](https://img.shields.io/github/languages/code-size/joshgc19/landscape_recognition_model)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)

# Table of contents
* [Landscape Detector](#landscape-detector)
  * [Table of contents](#table-of-contents)
  * [Installation](#installation)
  * [Code Structure](#code-structure)
  * [Implementation Overview](#implementation-overview)
    * [Design Constraints](#design-constraints)
    * [Chosen Features](#chosen-features)
    * [Chosen Classifier](#chosen-classifier)
  * [Data](#data)
    * [Target Data Requirements](#target-data-requirements) 
    * [Source Data](#source-data)
    * [Data Preprocessing](#data-preprocessing)
  * [Results](#results)
    * [Training](#training)
    * [Testing](#testing)
  * [Conclusion](#conclusion)
  * [Project Author](#project-author)
  * [License](#license)

# Installation and setup
To install needed dependencies for this project you can use the following pip command:
```bash
pip install requirements.txt
```
The project was implemented using python v3.12.2 and the latest libraries versions.

# Code Structure
```bash
├── data
│   ├── testing 
│   └── training
├── model_vectors
│   ├── individuals.txt
│   └── obtained_model.txt
├── preprocessed_data
│   ├── testing 
│   └── training
├── features_extraction.py
├── preprocessing.py
├── recognition.py
├── utils.py
├── main.py
├── requirements.txt
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

## Chosen Classifier
**The Gaussian Naive Bayes (GNB)** classifier was chosen for this project due to several key advantages that make it well-suited for the task at hand. First, the GNB classifier is particularly effective when dealing with continuous data that can be assumed to be normally distributed, which aligns with the nature of the features in our dataset. By modeling the landscape features as a Gaussian distribution, the GNB classifier can efficiently estimate the likelihood of an image being a target landscape or not based on its features' mean and variance.


# Data
## Target data requirements
  * The picture must have the following hierarchical structure from top to bottom:
    * Blue daylight sky at the top with few clouds.
    * Solid ground must be portrayed with dominance of nature at the middle.
    * Buildings and housing may be present at the middle-bottom level.
  * Bodies of water may not be visible.
  * Depicted nature must be alive or green.
  * No snow coverage must be shown.
## Source Data
All source images were obtained from Pexels.com, a platform offering a wide range of high-quality, free stock photos and videos. These images are commonly used for various projects and are licensed under the Creative Commons Zero (CC0) license, which means they are free for personal and commercial use without attribution required.

## Data Preprocessing
To easily extract the features from the dataset, from each feature an intermediate image was created, these were the following:

* **Green Color Mask**:A color mask was applied to the original image to extract only the green pixels.
* **Sky Color Mask (Light Blue)**: A color mask was applied to the original image to focus on the pixels representing the sky color, a light blue.
* **50-Stripe Gradient**: A 50-stripe gradient was created from top to bottom in the original image. Each stripe of the gradient was calculated using the average of the pixels in that stripe.



# Results
## Training
As a result of the training stage and each individual features vector, the average vector ( $E(\vec{F})$ ) which reflect central tendencies of the datas set and the variance vector ($\sigma_{\vec{F}}$) which represents the variability across the data set. Both resulting vector are shown down below:
    
$$E(\vec{F}) = (0.32, 0.3315, 0.2601)$$

$$\sigma_{\vec{F}} = (0.1841, 0.0540, 0.1428)$$
## Testing
The objetive of testing the current project is to determine the accuracy of the generated model. The testing dataset was composed of 10 high resolution images, which contained 5 landscape images and 5 non-landscape images such as animals or random settings.

Using the automated testing process within the recognition module the following results were found: 

| **Classification outcomes** | *Relative Frequency* |
|-----------------------------|:--------------------:|
| False positive              |         10%          |
| False negative              |         20%          |
| True positive               |         30%          |
| True negative               |         40%          |

Taking into account the true positive and true negative outcomes the accuracy level of the obtained model was of exactly 70%.

# Conclusion
As a summary, the preprocessing stage is critically important as it significantly impacts the quality of feature extraction, which forms the foundation of the model. Ensuring the correct selection of features and processing of data leads to a more precise model. Conversely, if the preprocessing stage is not carefully executed, the model may end up with a suboptimal set of features, leading to poor classification performance. Therefore, it is crucial to pay close attention to the preprocessing stage and features selection.

# Project author

## **Joshua Gamboa Calvo**<br>
BS in Computing Engineering undergraduate<br>
Instituto Tecnológico de Costa Rica<br>
<br>
[<img src="https://edent.github.io/SuperTinyIcons/images/svg/linkedin.svg" width="50px" style="border-radius:50%">](https://www.linkedin.com/in/joshgc19)  [<img src="https://edent.github.io/SuperTinyIcons/images/svg/github.svg" width="50px" style="border-radius:50%"/>](https://github.com/joshgc19)  [<img src="https://edent.github.io/SuperTinyIcons/images/svg/medium.svg" width="50px" style="border-radius:50%">](https://medium.com/@joshgc.19)<br> 


## License
>You can checkout the full license [here (opens in the same tab)](https://github.com/joshgc19/landscape_recognition_model/blob/master/LICENSE). 

This project is licensed under the terms of the **MIT** license. 