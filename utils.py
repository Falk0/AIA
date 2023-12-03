# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:16:33 2023

@author: Baumann
"""

import cv2
from cv2 import threshold, THRESH_BINARY
import numpy as np
from skimage.morphology import skeletonize
from numba import jit
from time import perf_counter

def thin_spiral_image_with_custom_cut(image_path:str, white_out:bool=True, field_of_interest:list=[(57, -81), (122, -91)]):
    # Load the image and convert it to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # White out the specified regions
    cut_img = img.copy()
    if white_out:
        cut_img[:field_of_interest[0][0], :] = 255  # Top region
        cut_img[field_of_interest[0][1]:, :] = 255  # Bottom region
        cut_img[:, :field_of_interest[1][0]] = 255  # Left region
        cut_img[:, field_of_interest[1][1]:] = 255  # Right region
    
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
    _, skeleton_uint8_bin = threshold(skeleton_uint8, 128, 1, THRESH_BINARY)
    return img, cut_img, skeleton_uint8_bin #skeleton_uint8

@jit(nopython=True)
def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

@jit(nopython=True)
def angle_func(a, center):
    return np.arctan((center[1] - a[1]) / (center[0] - a[0] + 1e-10))

@jit(nopython=True)
def get_angle_dist2(a:list):
    angle = np.zeros(len(a)-1)
    dist = np.zeros(len(a)-1)
    for i in range(len(a)-1):
        angle[i] = angle_func(a[i+1], a[0])  #np.arctan((a[0][1] - a[i+1][1]) / (a[0][0] - a[i+1][0] + 1e-10))
        dist[i] = dist2(a[i+1], a[0])  #(a[i+1][0]-a[0][0])**2 + (a[i+1][1]-a[0][1])**2
    return angle, dist

@jit(nopython=True)
def _choose_closest(imgFiltered, center, search_radius):
    middle = [-1, -1]
    for j in [11, 13, 12]: # 11 checks for end points, 13 checks for intersections, 12 checks for normal connections
        
        #ends = np.argwhere(imgFiltered == j)
        # np.argwhere cannot be used with numba because output needs to be list of lists and not list of tuples
        ends = []
        for ix, x in enumerate(imgFiltered):
            for iy in range(len(x)):
                if imgFiltered[ix][iy]==j:
                    ends.append([ix, iy])
        
        for i in ends:
            if dist2(i, center) < dist2(middle, center):
                middle = i
        if dist2(middle, center) < search_radius**2:
            return (middle[1], middle[0])
    return None

def find_spiral_point(skeleton_image, point:tuple=(0, 0), first:bool=True, search_radius:int=200):
    # Find the center of the image
    if first:
        center_x, center_y = skeleton_image.shape[1] // 2, skeleton_image.shape[0] // 2
    else:
        center_x, center_y = point[0], point[1]
    center = (center_y, center_x)    
    
    # Set the end-points kernel:
    h = np.array([[1, 1, 1],
                  [1, 10, 1],
                  [1, 1, 1]])
    # Convolve the image with the kernel:
    imgFiltered = cv2.filter2D(skeleton_image, -1, h)
    return _choose_closest(imgFiltered, center, search_radius)

@jit(nopython=True)
def find_spiral_neigbor(skeleton_image, point=[0, 0], first:bool=True, search_radius=200):
    # Find the center of the image
    if first:
        center_x, center_y = skeleton_image.shape[1] // 2, skeleton_image.shape[0] // 2
    else:
        center_x, center_y = point[0], point[1]
        
    # Search for the start of the spiral within the search radius
    for radius in range(1, search_radius + 1):
        for angle in np.linspace(0, 2 * np.pi, 8 * radius): #8 * radius
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

            if 0 <= x < skeleton_image.shape[1] and 0 <= y < skeleton_image.shape[0]:
                if skeleton_image[y, x] == 1:  # White pixel found
                    return x, y  # Returning the first white pixel found in spiral pattern
    return None  # Return None if no start found

def trace_spiral(skeleton_image, start_point:tuple, search_radius=200):
    current_point = start_point
    #path = np.zeros()
    path = list([current_point])

    while True:
        # Remove the current point from the skeleton image
        skeleton_image[current_point[1], current_point[0]] = 0

        # Find the nearest neighbor in the local region
        next_point = find_spiral_neigbor(skeleton_image, (current_point[0], current_point[1]), False, search_radius) #find_local_nearest_neighbor(skeleton_image, current_point)
        
        if next_point is None:
            break  # No more neighbors

        # Update the current point
        current_point = next_point
        path.append(current_point)
        
    return path

def angle_cont_func(trace_angle_continuous):
    ''' 
    takes a list of lists of angles
    returns the list of lists again but with a continuoous representation
    '''
    for index, j in enumerate(trace_angle_continuous):
        for i in range(len(j)-1):
            if j[i] - j[i+1] > 2:
                trace_angle_continuous[index][i+1:] += np.pi * np.ones(len(j[i+1:]))
            elif j[i] - j[i+1] < -2:
                trace_angle_continuous[index][i+1:] -= np.pi * np.ones(len(j[i+1:]))
    return trace_angle_continuous
