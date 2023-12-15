# Spiral Image Processing Project
### Description

This project focuses on processing spiral images to achieve a thinned, skeletonized representation of the spiral structure. 
It involves image manipulation techniques such as cropping, binary thresholding, skeletonization, and morphological operations. 
The goal is to trace the spiral structure pixel by pixel, calculate distances from a starting point, and potentially visualize this path.
In the end the extracted distance is compared to a template (which in this case is a smoothed version of the distance).
The standard deviation of this difference is then used as input to a classifier.

### Features

    Image Cropping and Modification: Adjusts the input image by cropping or whiting out specific regions.
    Binary Thresholding: Converts the modified image to a binary format for further processing.
    Skeletonization: Thins the spiral to a one-pixel width line, making it easier to trace and analyze.
    Morphological Closing (optional): Attempts to close small gaps in the skeletonized image using morphological operations.
    Spiral Tracing and Distance Calculation: Traces the spiral from a specified starting point and calculates the distances of each pixel from this point.

### Requirements
see: [requirements.txt](requirements.txt)

    Python 3.x
    OpenCV (opencv-python)
    NumPy (numpy)
    SciPy (scipy)
    scikit-image (scikit-image)
    Numba (numba)

### Usage

    Prepare the Image: Load your spiral image using OpenCV.
    Modify the Image: Crop or white out specific regions as needed.
    Threshold and Skeletonize: Convert the image to binary and then apply the skeletonization process.
    Apply Morphological Closing (Optional): Use if there are small gaps in the skeleton that need to be closed.
    Trace the Spiral: Implement the spiral tracing algorithm starting from a specific point.
    Analyze the Spiral: Calculate and plot the distances from the starting point along the spiral.

[PD_classification_Kamble.ipynb](PD_classification_Kamble.ipynb), [PD_classification_Varalakshmi.ipynb](PD_classification_Varalakshmi.ipynb) and [PrimSec_Classification_Studium.ipynb](PrimSec_Classification_Studium.ipynb) contain a complete toolchain for three different datasets.

[utils.py](utils.py) contains helpful functions for all datasets.

[PD_classification_Kamble_test_features.ipynb](PD_classification_Kamble_test_features.ipynb) contains some experimental functions.

### Note

This project includes experimental image processing techniques. Results may vary based on the input image characteristics and the chosen parameters.
