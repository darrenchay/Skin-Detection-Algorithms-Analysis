""" 
Name: Darren Chay Loong
ID: 1049254
Course: CIS*4720
Assignment 3

Description: This file prompts the user to choose an image file which will be checked to see if their is any
             skin present in the image using one of the 3 algorithms implemented below. After choosing the image, 
             the user will be prompted to choose which algorithm will be used and the resulting image will be stored
             in the Results folder

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

# This algorithm by Peer et al. uses the RGB color space to detect skin based on several masks 
def peerAlgo(img):
    # Convert from BGR to RGB format
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pprint.pprint(imgRGB)
    filteredImg = np.zeros((rows, cols), dtype=np.float32)

    # Split the R, G, & B channels
    rChannel, gChannel, bChannel = cv.split(imgRGB)
    # print(rChannel, bChannel, gChannel)
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
        maskR4[rChannel > bChannel] = 255
        maskR4[rChannel <= bChannel] = 0
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
    return filteredImg


# def cvtRGB2HSI(img):
#     # pre_proc
#     rows, cols = img.shape[0], img.shape[1]
#     # transpose to 3 * pixels_number
#     # utilized for extract R/G/B cols
#     t_p = img.swapaxes(0, 2).swapaxes(1, 2)
#     R, G, B = t_p[0], t_p[1], t_p[2]
#     dRG, dRB, dGB = R - G, R - B, G - B  # all belongs to [0,255]
#     dRG[dRG == 0.0] = 0.5
#     dRB[dRB == 0.0] = 0.5
#     dGB[dGB == 0.0] = 0.5  # just ignore my divide_zero exception handler
#     # H part
#     cos = (dRG+dRB)/(2*np.sqrt(dRG**2+dRB*dGB))
#     H = np.arccos(cos)
#     H[np.isnan(H)] = 0.0  # if cos is too big, arccos returns nan
#     # I part
#     I = img.mean(axis=2)
#     # prepare for Saturation calc
#     Imin = img.min(axis=2)
#     I[I == 0.0] = 1
#     # S part
#     S = 1 - Imin / I
#     return H,S,I


# def linAlgo(img):

#     # Convert from BGR to HSI format
#     img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
#     filteredImg = np.zeros((rows, cols), dtype=np.float32)

#     H, S, I  = cvtRGB2HSI(img)
#     # Split the H, S, & I channels
#     hChannel, sChannel, iChannel = cv.split(img)
#     print(hChannel, sChannel, iChannel)
#     # Getting max and mins of each channel
#     maxH = int(np.max(hChannel))
#     minH = int(np.min(hChannel))

#     maxS = int(np.max(sChannel))
#     minS = int(np.min(sChannel))
#     print("maxS", maxS)

#     maxI = int(np.max(iChannel))
#     minI = int(np.min(iChannel))

#     # Creating R1 Mask (I > 40)
#     lowerBound = np.array((minH, minS, 40))
#     upperBound = np.array((maxH, maxS, maxI))
#     print("Bounds")
#     # pprint.pprint(lowerBound)
#     maskR1 = cv.inRange(img, lowerBound, upperBound)
#     print("Mask R1")
#     pprint.pprint(maskR1)

#     # Creating R2 part a Mask (13 < S < 110 & 0° < H < 28°)
#     lowerBound = np.array((0, 13, minI))
#     upperBound = np.array((28, 110, maxI))
#     maskR2_a = cv.inRange(img, lowerBound, upperBound)
#     # print("Mask R2")
#     # pprint.pprint(maskR2)

#     # Creating R2 part b Mask (13 < S < 110 & 332° < H < 360°)
#     lowerBound = np.array((332, 13, minI))
#     upperBound = np.array((360, 110, maxI))
#     maskR2_b = cv.inRange(img, lowerBound, upperBound)
#     # print("Mask R3")
#     # pprint.pprint(maskR3)

#     # Creating R4 Mask (13 < S < 75 & 309° < H < 331°)
#     lowerBound = np.array((309, 13, minI))
#     upperBound = np.array((331, 75, maxI))
#     maskR3 = cv.inRange(img, lowerBound, upperBound)
#     # print("Mask R3")
#     # pprint.pprint(maskR3)

#     # Applying all the masks to the image
#     filteredImg = cv.bitwise_and(img, img, mask=maskR1)
#     maskR2 = cv.bitwise_or(maskR2_a, maskR2_b)
#     filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
#     filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR3)

#     plt.subplot(1, 5, 1)
#     plt.imshow(maskR1, cmap="gray")
#     plt.subplot(1, 5, 2)
#     plt.imshow(maskR2_a, cmap="gray")
#     plt.subplot(1, 5, 3)
#     plt.imshow(maskR2_b, cmap="gray")
#     plt.subplot(1, 5, 4)
#     plt.imshow(maskR3, cmap="gray")
#     plt.subplot(1, 5, 5)
#     plt.imshow(filteredImg)
#     plt.show()

#     filteredImg = cv.cvtColor(filteredImg, cv.COLOR_BGR2RGB)
#     return filteredImg

# This algorithm by Chai et al. uses the YCrCb color space to detect skin based on several masks 
def chaiAlgo(img):
    # Convert from BGR to YCbCr colorspace
    imgYCbCr = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    filteredImg = np.zeros((rows, cols))

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

    # Applying masks to image
    filteredImg = cv.bitwise_and(imgYCbCr, imgYCbCr, mask=maskR1)
    filteredImg = cv.bitwise_and(filteredImg, filteredImg, mask=maskR2)
    filteredImg = cv.cvtColor(filteredImg, cv.COLOR_YCR_CB2BGR)

    # Converting all green pixels to black
    filteredImg[np.all(filteredImg == (0, 135, 0), axis=-1)] = (0, 0, 0)
    return filteredImg

# This algorithm by Wang & Yuan uses the HSV and normalized RGB color space to detect skin based on several masks from both color spaces
def wangAlgo(img):
    # Convert from BGR to HSV colorspace
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Creating R1 Mask (0 < H < 25, 51 < S < 173, 89 < v < 255)
    lowerBound = np.array((0, 51, 89))
    upperBound = np.array((25, 173, 255))
    maskR1 = cv.inRange(imgHSV, lowerBound, upperBound)
    # cv.imwrite("maskR1.png", maskR1)
    
    # Creating R2 Mask (0.36 < r < 0.465 & 0.28 < g < 0.363)
    imgRGB = normalized(img)
    lowerBound = np.array((0, 71, 92))
    upperBound = np.array((255, 93, 119))
    maskR2 = cv.inRange(imgRGB, lowerBound, upperBound)
    # cv.imwrite("maskR2.png", maskR2)
    
    # Applying mask R1
    filteredImgHSV = cv.bitwise_and(imgHSV, imgHSV, mask=maskR1)
    # cv.imwrite("HSV.png", filteredImgHSV)
    # Converting back to BGR color space
    filteredImgRGB = cv.cvtColor(filteredImgHSV, cv.COLOR_HSV2BGR)
    # Apply mask R2
    filteredImgRGB = cv.bitwise_and(filteredImgRGB, filteredImgRGB, mask=maskR2)
    
    return filteredImgRGB


# This function converts an image in the RGB color space to the normalized RGB colorspace (rg chrominance)
def normalized(img):
        # Creating the normalized img
        norm=np.zeros((img.shape[0],img.shape[1],3),np.float32)
        norm_rgb=np.zeros((img.shape[0],img.shape[1],3),np.uint8)

        # Getting the 3 color channels
        b=img[:,:,0]
        g=img[:,:,1]
        r=img[:,:,2]

        # Getting the sum of RGB values and storing them as ints
        sum=b.astype(int) + g.astype(int) + r.astype(int)
        # pprint(sum)

        # Normalizing and scaling values of rgb to 255
        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0

        # Normalizing to rgb
        norm_rgb=cv.convertScaleAbs(norm)
        # pprint.pprint(norm_rgb)
        # cv.imwrite("normalized.png", norm_rgb)
        return norm_rgb
  
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

# Choosing which algorithm to run
algorithmNo = 0
while int(algorithmNo) not in [1, 2, 3]:
    print("******************************************")
    print("\t\tAlgorithms:")
    print("******************************************")
    print("Peer et al. (RGB color space):\t\t", 1)
    print("Chai & Ngan (YCbCr color space):\t", 2)
    print("Wang & Yuan (HSV & normalized RGB):\t", 3)
    # print("Lin et al. (HSI color space):\t\t", 4)
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

    print("=========================\nRunning Peer et al. Algorithm\n=========================")
    processedImage = peerAlgo(img)
elif algorithmNo is 2:
    newFileName = 'Results/chai-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("=========================\nRunning Chai & Ngan Algorithm\n=========================")
    processedImage = chaiAlgo(img)
elif algorithmNo is 3:
    newFileName = 'Results/wang-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("=========================\nRunning Wang & Yuan Algorithm\n=========================")
    processedImage = wangAlgo(img)
else:
    newFileName = 'Results/lin-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("=========================\nRunning Lin et al. Algorithm\n=========================")
    # processedImage = linAlgo(img)

# Ending timer
endTime = time.time()
timeTaken = round(endTime - startTime, 3)
print("Time taken: ", timeTaken, " seconds")

# Saving processed image
cv.imwrite(newFileName, processedImage)
