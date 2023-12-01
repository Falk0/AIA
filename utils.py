# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:16:33 2023

@author: Baumann
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

def thin_spiral_image_with_custom_cut(image_path:str, white_out:bool=True, field_of_interest:list=[(-81, 57), (-91, 122)]):
    # Load the image and convert it to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # White out the specified regions
    cut_img = img.copy()
    if white_out:
        cut_img[:field_of_interest[0][1], :] = 255  # Top region
        cut_img[field_of_interest[0][0]:, :] = 255  # Bottom region
        cut_img[:, :field_of_interest[1][1]] = 255  # Left region
        cut_img[:, field_of_interest[1][0]:] = 255  # Right region
    
    # Set all non-white pixels to black
    binary_strict = np.where(cut_img < 255, 0, 255).astype(np.uint8)

    # Invert the binary image for skeletonization
    binary_inverted_strict = cv2.bitwise_not(binary_strict)

    # Skeletonize the image
    skeleton = skeletonize(binary_inverted_strict // 255) * 255
    skeleton_uint8 = skeleton.astype(np.uint8)

    # Closing gaps in skeleton
    #skeleton_uint8 = closing(skeleton_uint8, square(3))

    # Return the cut and skeletonized image for plotting
    return img, cut_img, skeleton_uint8 #skeleton_uint8

def dist2(a:tuple, b=tuple):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def angle_func(a:tuple, center=tuple):
    return np.arctan((center[1] - a[1]) / (center[0] - a[0] + 1e-10))

def find_spiral_start(skeleton_image, point:tuple=(0, 0), first:bool=True):
    # Find the center of the image
    if first:
        center_x, center_y = skeleton_image.shape[1] // 2, skeleton_image.shape[0] // 2
    else:
        center_x, center_y = point[0], point[1]
    center = (center_y, center_x)
    # Define a larger search radius
    search_radius = 200
   
    _, binaryImage = cv2.threshold(skeleton_image, 128, 1, cv2.THRESH_BINARY)
    # Set the end-points kernel:
    h = np.array([[1, 1, 1],
                  [1, 10, 1],
                  [1, 1, 1]])
    # Convolve the image with the kernel:
    imgFiltered = cv2.filter2D(binaryImage, -1, h)
    
    nearest_of_all = (0, 0)
    for j in [11, 13, 12]: # 11 checks for end points, 13 checks for intersections, 12 checks for normal connections
        #print(j)
        endPointsMask = np.where(imgFiltered == j, 255, 0)
        endPointsMask_8 = endPointsMask.astype(np.uint8)
        ends = np.argwhere(endPointsMask_8==255)
        middle = (0, 0)
        for i in ends:
            if dist2(i, center) < dist2(middle, center):
                middle = tuple(i)
                #print(middle)
        
        if dist2(nearest_of_all, center) < dist2(middle, center):
            nearest_of_all = middle
        #print(dist2(middle, center))
        if dist2(middle, center) < search_radius**2:
            return (middle[1], middle[0])
        
    return (nearest_of_all[1], nearest_of_all[0])
    '''
    # Search for the start of the spiral within the search radius
    for radius in range(1, search_radius + 1):
        for angle in np.linspace(0, 2 * np.pi, 8 * radius):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

            if 0 <= x < skeleton_image.shape[1] and 0 <= y < skeleton_image.shape[0]:
                if skeleton_image[y, x] == 255:  # White pixel found
                    return x, y  # Returning the first white pixel found in spiral pattern

    return None  # Return None if no start found
    '''

def find_spiral_neigbor(skeleton_image, point:tuple=(0, 0), first:bool=True):
    # Find the center of the image
    if first:
        center_x, center_y = skeleton_image.shape[1] // 2, skeleton_image.shape[0] // 2
    else:
        center_x, center_y = point[0], point[1]
        
    # Define a larger search radius
    search_radius = 200
    
    # Search for the start of the spiral within the search radius
    for radius in range(1, search_radius + 1):
        for angle in np.linspace(0, 2 * np.pi, 8 * radius):
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

            if 0 <= x < skeleton_image.shape[1] and 0 <= y < skeleton_image.shape[0]:
                if skeleton_image[y, x] == 255:  # White pixel found
                    return x, y  # Returning the first white pixel found in spiral pattern

    return None  # Return None if no start found
    

def trace_spiral_and_create_image(skeleton_image, start_point):
    # Create a blank RGB image for visualization
    trace_image = np.zeros((skeleton_image.shape[0], skeleton_image.shape[1], 3), dtype=np.uint8)

    current_point = start_point
    path = [current_point]
    distances = [0.0]
    angles = []

    # Set the starting point in the trace image
    trace_image[current_point[1]-5:current_point[1]+5, current_point[0]-5:current_point[0]+5] = [255, 0, 0]  # Red for starting point
    
    #counter = 0
    while True:
        # Remove the current point from the skeleton image
        skeleton_image[current_point[1], current_point[0]] = 0

        # Mark the current point in the trace image with a different color
        trace_image[current_point[1], current_point[0]] = [0, 255, 0]  # Green for traced path

        # Find the nearest neighbor in the local region
        next_point = find_spiral_neigbor(skeleton_image, (current_point[0], current_point[1]), False) #find_local_nearest_neighbor(skeleton_image, current_point)
        if next_point is None:
            break  # No more neighbors

        # Calculate the distance from the start
        distance = np.sqrt(dist2(next_point, start_point))
        distances.append(distance)
        
        # calculate the angle from the start
        angle = angle_func(next_point, center=start_point)
        angles.append(angle)

        # Update the current point
        current_point = next_point
        path.append(current_point)
        
        #counter += 1
    return path, distances, trace_image, angles
