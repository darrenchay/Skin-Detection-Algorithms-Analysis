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
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    filteredImg = np.zeros((rows, cols), dtype=np.float32)

    # Split the R, G, & B channels
    rChannel, gChannel, bChannel = cv.split(imgRGB)
    print(rChannel, bChannel, gChannel)
    if (np.max(imgRGB) - np.min(imgRGB)) > 15:
        
        # Creating R1 Mask (R > 95 and G > 40 and B > 20)
        lowerBound = np.array((95, 40, 20))
        upperBound = np.array((255, 255, 255))
        maskR1 = cv.inRange(imgRGB, lowerBound, upperBound)
        # print("Mask R1")
        # pprint.pprint(maskR1)
        
        # Creating R2 Mask (|R−G| > 15)
        maskR2 = rChannel - gChannel
        maskR2[abs(rChannel - gChannel) > 15] = 255
        maskR2[abs(rChannel - gChannel) <= 15] = 0
        # print("Mask R2")
        # pprint.pprint(maskR2)
        
        # Creating R3 Mask (R > G)
        maskR3 = rChannel - gChannel
        maskR3[rChannel > gChannel] = 255
        maskR3[rChannel <= gChannel] = 0
        # print("Mask R3")
        # pprint.pprint(maskR3)
        
        # Creating R4 Mask (R > B)
        maskR4 = rChannel - bChannel
        maskR4[ rChannel > bChannel] = 255
        maskR4[ rChannel <= bChannel] = 0
        # print("Mask R4")
        # pprint.pprint(maskR4)
        
        # Applying all the masks to the image
        filteredImg = cv.bitwise_and(imgRGB, imgRGB, mask=maskR1)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR3)
        filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR4)
        
        filteredImg = cv.cvtColor(filteredImg, cv.COLOR_BGR2RGB)
        # plt.subplot(1, 5, 1)
        # plt.imshow(maskR1, cmap="gray")
        # plt.subplot(1, 5, 2)
        # plt.imshow(maskR2, cmap="gray")
        # plt.subplot(1, 5, 3)
        # plt.imshow(maskR3, cmap="gray")
        # plt.subplot(1, 5, 4)
        # plt.imshow(maskR4, cmap="gray")
        # plt.subplot(1, 5, 5)
        # plt.imshow(filteredImg)
        
        # # Saving processed image
        # newFileName = 'Results/rgbColorSpace' + \
        # filenameSplit[len(filenameSplit)-1]
        # cv.imwrite(newFileName, filteredImg)
        # plt.show()
    return filteredImg
        
    
def linAlgo(img):
    # Convert from BGR to HSI format
    img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    filteredImg = np.zeros((rows, cols), dtype=np.float32)

    # Split the H, S, & I channels
    hChannel, sChannel, iChannel = cv.split(img)
    print(hChannel, sChannel, iChannel)
    # Getting max and mins of each channel
    maxH = int(np.max(hChannel))
    minH = int(np.min(hChannel))
    
    maxS = int(np.max(sChannel))
    minS = int(np.min(sChannel))
    print("maxS", maxS)
    
    maxI = int(np.max(iChannel))
    minI = int(np.min(iChannel))
    
    # Creating R1 Mask (I > 40)
    lowerBound = np.array((minH, minS, 40))
    upperBound = np.array((maxH, maxS, maxI))
    print("Bounds")
    # pprint.pprint(lowerBound)
    maskR1 = cv.inRange(img, lowerBound, upperBound)
    print("Mask R1")
    pprint.pprint(maskR1)
    
    # Creating R2 part a Mask (13 < S < 110 & 0° < H < 28°)
    lowerBound = np.array((0, 13, minI))
    upperBound = np.array((28, 110, maxI))
    maskR2_a = cv.inRange(img, lowerBound, upperBound)
    # print("Mask R2")
    # pprint.pprint(maskR2)
    
    # Creating R2 part b Mask (13 < S < 110 & 332° < H < 360°)
    lowerBound = np.array((332, 13, minI))
    upperBound = np.array((360, 110, maxI))
    maskR2_b = cv.inRange(img, lowerBound, upperBound)
    # print("Mask R3")
    # pprint.pprint(maskR3)
    
    # Creating R4 Mask (13 < S < 75 & 309° < H < 331°)
    lowerBound = np.array((309, 13, minI))
    upperBound = np.array((331, 75, maxI))
    maskR3 = cv.inRange(img, lowerBound, upperBound)
    # print("Mask R3")
    # pprint.pprint(maskR3)
    
    # Applying all the masks to the image
    filteredImg = cv.bitwise_and(img, img, mask=maskR1)
    maskR2 = cv.bitwise_or(maskR2_a, maskR2_b)
    filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
    filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR3)
    
    plt.subplot(1, 5, 1)
    plt.imshow(maskR1, cmap="gray")
    plt.subplot(1, 5, 2)
    plt.imshow(maskR2_a, cmap="gray")
    plt.subplot(1, 5, 3)
    plt.imshow(maskR2_b, cmap="gray")
    plt.subplot(1, 5, 4)
    plt.imshow(maskR3, cmap="gray")
    plt.subplot(1, 5, 5)
    plt.imshow(filteredImg)
    plt.show()
    
    filteredImg = cv.cvtColor(filteredImg, cv.COLOR_BGR2RGB)
    return filteredImg
        
        
def chaiAlgo(img):
    # Convert from BGR to YCbCr colorspace
    imgYCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    filteredImg = np.zeros((rows, cols))

    # Split the Y, Cb, & Cr channels
    # yChannel, CrChannel, CbChannel = cv.split(img)
    # pprint.pprint(yChannel)
    # pprint.pprint(CrChannel)
    # pprint.pprint(CbChannel)
    # print(np.max(yChannel), np.max(CbChannel), np.max(CrChannel))
        
    # Creating R1 Mask (77 ≤ Cb ≤ 127)
    lowerBound = np.array((0, 0, 77))
    upperBound = np.array((255, 255, 127))
    maskR1 = cv.inRange(imgYCbCr, lowerBound, upperBound)
    # print("Mask R1")
    # pprint.pprint(maskR1)   
    
    # Creating R2 Mask (133 ≤ Cr ≤ 173)
    lowerBound = np.array((0, 133, 0))
    upperBound = np.array((255, 173, 255))
    maskR2 = cv.inRange(imgYCbCr, lowerBound, upperBound)    
    
    filteredImg = cv.bitwise_and(imgYCbCr, imgYCbCr, mask=maskR1)
    filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
    filteredImg = cv.cvtColor(filteredImg, cv.COLOR_YCR_CB2BGR)
    
    # Post processing (removing green background)
    bChannel, gChannel, rChannel = cv.split(filteredImg)
    filteredImg
    tempMask = cv.inRange(filteredImg, lowerBound, upperBound)    
    filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=tempMask)
    # filteredImg = cv.cvtColor(filteredImg, cv.COLOR_BGR2RGB)
    return filteredImg
        
        
# Reading the file
Tk().withdraw()
filename = askopenfilename()
# print("filename: ", filename)

# Check if file was selected
if not filename:
    print("error: no file selected")
    exit(1)

# Reading Image
img = cv.imread(filename, -1)  
# img = np.float32(img)/255
pprint.pprint(img.shape)
rows, cols, extra = img.shape
# peerAlgo(img)

# Choosing which algorithm to run
algorithmNo = 0
while int(algorithmNo) not in [1, 2, 3, 4]:
    print("******************************************")
    print("\t\tAlgorithms:")
    print("******************************************")
    print("Peer et al. (RGB color space):\t\t", 1)
    print("Lin et al. (HSI color space):\t\t", 2)
    print("Chai & Ngan (YCbCr color space):\t", 3)
    print("Mode Filter:\t\t", 4)
    print("******************************************")
    algorithmNo = input("Enter the number of the algorithm you want to run: ")
    if not algorithmNo.isnumeric():
        algorithmNo = 0

algorithmNo = int(algorithmNo)

# Saving the name of the output image file
filenameSplit = filename.split("/")

# Starting timer
startTime = time.time()

# Running the appropriate algorithm
if algorithmNo is 1:
    newFileName = 'Results/peer-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Peer et al. Algorithm\n====================")
    processedImage = peerAlgo(img)
elif algorithmNo is 2:

    newFileName = 'Results/lin-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Lin et al. Algorithm\n====================")
    processedImage = linAlgo(img)
elif algorithmNo is 3:

    newFileName = 'Results/chai-output' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Chai & Ngan Algorithm\n====================")
    processedImage = chaiAlgo(img)
else:
    newFileName = 'Results/nagao-matsuyama_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Nagao-Matsuyama Filter\n====================")
    # processedImage = nagaoMatsuyamaFilter(img)

# Ending timer
endTime = time.time()
timeTaken = round(endTime - startTime, 3)
print("Time taken: ", timeTaken, " seconds")

# Saving processed image
cv.imwrite(newFileName, processedImage)