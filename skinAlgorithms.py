""" 
Name: Darren Chay Loong
ID: 1049254
Course: CIS*4720
Assignment 3

Description: 
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

# ALGORITHMS


def peerAlgo(img):
    # Convert from BGR to RGB format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    filteredImg = np.zeros((rows, cols), dtype=np.float32)
    rChannel, gChannel, bChannel = cv.split(img)
    print(rChannel[0], bChannel[0], gChannel[0])
    if (np.max(img) - np.min(img)) > 15:
        # Finding R1 Mask
        lowerBound = np.array((95, 40, 20))
        upperBound = np.array((255, 255, 255))
        maskR1 = cv.inRange(img, lowerBound, upperBound)
        print("Mask R1")
        pprint.pprint(maskR1)
        filteredImg = cv.bitwise_and(img, img, mask=maskR1)
        
        # Finding R2 Mask
        maskR2 = rChannel - gChannel
        maskR2[abs(rChannel - gChannel) > 15] = 255
        maskR2[abs(rChannel - gChannel) <= 15] = 0
        print("Mask R2")
        pprint.pprint(maskR2)
        # pprint.pprint(maskR2)
        
        # Finding R3 Mask
        maskR3 = rChannel - gChannel
        maskR3[rChannel > gChannel] = 255
        maskR3[rChannel <= gChannel] = 0
        print("Mask R3")
        pprint.pprint(maskR3)
        
        # Finding R4 Mask
        maskR4 = rChannel - bChannel
        pprint.pprint(maskR4)
        maskR4[ rChannel > bChannel] = 255
        maskR4[ rChannel <= bChannel] = 0
        print("Mask R4")
        pprint.pprint(maskR4)
        
        # finalMask = maskR1 + maskR1 + maskR3 + maskR4
        filteredImg = cv.bitwise_and(img, img, mask=maskR1)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR3)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR4)
        
        
        plt.subplot(1, 5, 1)
        plt.imshow(maskR1, cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(maskR2, cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(maskR3, cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(maskR4, cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(filteredImg)
        
        filteredImg = cv.cvtColor(filteredImg, cv.COLOR_BGR2RGB)
        # Saving processed image
        newFileName = 'Results/rgbColorSpace' + \
        filenameSplit[len(filenameSplit)-1]
        cv.imwrite(newFileName, filteredImg)
        plt.show()
    
        
        
        
        
        
        
# Reading the file
Tk().withdraw()
filename = askopenfilename()
print("filename: ", filename)
filenameSplit = filename.split("/")

img = cv.imread(filename, -1)  # reading image 
# img = np.float32(img)/255
pprint.pprint(img.shape)
rows, cols, extra = img.shape
peerAlgo(img)

