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

### Constraints

## Algorithm explanation

### Features chosen
To fully understand the features chosen in the current project, the following features of the type of landscapes that the model will recognize have to be reviewed:
  
  * The picture must have the following hierarchical structure from top to bottom:
    * Blue daylight sky at the top with few clouds.
    * Solid ground must be portrayed with dominance of nature at the middle.
    * Buildings and housing may be present at the middle-bottom level.
  * Bodies of water may not be visible.
  * Depicted nature must be alive or green.
  * No snow coverage must be shown.

Having the characterization of the image detailed, the features chosen to be part of the features vector used throughout the project will be the following:
  * **Sky percentage (SP)**: This feature represents the percentage of the image in which the sky is depicted. For this purpose a 50 stripes gradient must be calculated from top to bottom in search of a delta greater than 50 in _RGB_ scale.
  * **Greeness (G)**: Percentage of green pixels contained in the image, for this project the color green will have the following color bounds: $G_L = (25, 25, 20)$ y $G_U =  (230, 255, 86)$.
  * **Skyness (S)**:

### Pattern recognition and model calculation algorithm

### Implementation

## Resulting model

## Testing

### Results

## Conclusion

## Author

**Joshua Gamboa Calvo**<br>
BS in Computing Engineering undergraduate<br>
Instituto Tecnológico de Costa Rica<br>
[LinkedIn](https://www.linkedin.com/in/joshgc19) | [GitHub](https://github.com/joshgc.19)

## License

MIT License

Copyright (c) 2024 Joshua Gamboa Calvo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
