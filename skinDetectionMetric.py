""" 
Name: Darren Chay Loong
ID: 1049254
Course: CIS*4720
Assignment 3

Description: This file helps measure the percentage of true and false positives that are present 
             in the processed images by using a ground truth image as basis and comparing it to 
             the processed images in the form of a percentage

"""
# Image Processing Libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# Utility Libraries
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pprint
import time

print("Select the ground truth image")

# Extracting the skin section from the ground truth images
# Reading the ground truth file
Tk().withdraw()
filename = askopenfilename()
print("filename: ", filename)

# Check if file was selected
if not filename:
    print("error: no file selected")
    exit(1)

# Reading Image
grndTruthImg = cv.imread(filename, -1)
# img = np.float32(img)/255
print("Image dimensions: ")
pprint.pprint(grndTruthImg.shape)
rows, cols, extra = grndTruthImg.shape


# Creating the mask from the ground truth image
lowerBound = np.array((0, 0, 0))
upperBound = np.array((235, 235, 255)) # Removing all white-ish pixels from ground truth image
mask = cv.inRange(grndTruthImg, lowerBound, upperBound)
maskedTruthImg = cv.bitwise_and(grndTruthImg, grndTruthImg, mask=mask)
# cv.imwrite("testingGrndTruth.png", maskedTruthImg)

# Saving the count of pixels that are skin pixels and not skin pixels
skinPixelCount = np.count_nonzero(np.all(maskedTruthImg != (0, 0, 0), axis=-1))
print("Skin pixels: ",skinPixelCount)
notSkinPixelCount = (rows*cols) - skinPixelCount
print("Not skin pixels: ", notSkinPixelCount)


print("Select processed image")
# Comparing ground truth with mask

# Retreiving processed image
Tk().withdraw()
filename = askopenfilename()
print("filename: ", filename)

# Check if file was selected
if not filename:
    print("error: no file selected")
    exit(1)

# Reading processed image
skinImg = cv.imread(filename, -1)

# Applying ground truth mask to processed image
processedMaskedImg = cv.bitwise_and(skinImg, skinImg, mask=mask)
# cv.imwrite("testingProcessedImgMask.png", processedMaskedImg)

# Counting num of pixels that are stil present
processedSkinImgPixelCount = np.count_nonzero(np.all(processedMaskedImg != (0, 0, 0), axis=-1))
print("Skin pixels present in processed image:", processedSkinImgPixelCount)
# print("Not skin pixels: ", np.count_nonzero(np.all(processedMaskedImg == (0, 0, 0), axis=-1)))

print("True positive %: ", round(processedSkinImgPixelCount/skinPixelCount * 100, 3))


# Extracting false positive
falsePosImg = cv.bitwise_xor(skinImg, processedMaskedImg)
# cv.imwrite("testingFP.png", falsePosImg)

# Counting false positive pixels in processed image
falsePosSkinCount = np.count_nonzero(np.all(falsePosImg != (0, 0, 0), axis=-1))
print("Non skin pixels present in processed image:", falsePosSkinCount)
# print("Not skin pixels: ", np.count_nonzero(np.all(falsePosImg == (0, 0, 0), axis=-1)))

print("False positive %: ", round(falsePosSkinCount/notSkinPixelCount * 100, 3))


