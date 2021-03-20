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

        # # Saving processed image
        # newFileName = 'Results/rgbColorSpace' + \
        # filenameSplit[len(filenameSplit)-1]
        # cv.imwrite(newFileName, filteredImg)
        # plt.show()
    return filteredImg


def cvtRGB2HSI(img):
    # pre_proc
    rows, cols = img.shape[0], img.shape[1]
    # transpose to 3 * pixels_number
    # utilized for extract R/G/B cols
    t_p = img.swapaxes(0, 2).swapaxes(1, 2)
    R, G, B = t_p[0], t_p[1], t_p[2]
    dRG, dRB, dGB = R - G, R - B, G - B  # all belongs to [0,255]
    dRG[dRG == 0.0] = 0.5
    dRB[dRB == 0.0] = 0.5
    dGB[dGB == 0.0] = 0.5  # just ignore my divide_zero exception handler
    # H part
    cos = (dRG+dRB)/(2*np.sqrt(dRG**2+dRB*dGB))
    H = np.arccos(cos)
    H[np.isnan(H)] = 0.0  # if cos is too big, arccos returns nan
    # I part
    I = img.mean(axis=2)
    # prepare for Saturation calc
    Imin = img.min(axis=2)
    I[I == 0.0] = 1
    # S part
    S = 1 - Imin / I
    return H,S,I


def linAlgo(img):

    # Convert from BGR to HSI format
    img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    filteredImg = np.zeros((rows, cols), dtype=np.float32)

    H, S, I  = cvtRGB2HSI(img)
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
    # cv.imwrite("testingYCbCr.png", imgYCbCr)
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

    # Converting all green pixels to black
    filteredImg[np.all(filteredImg == (0, 135, 0), axis=-1)] = (0, 0, 0)
    return filteredImg

def wangAlgo(img):
    # Convert from BGR to HSV colorspace
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Creating R1 Mask (0 < H < 25, 51 < S < 173, 89 < v < 255)
    lowerBound = np.array((0, 51, 89))
    upperBound = np.array((25, 173, 255))
    maskR1 = cv.inRange(imgHSV, lowerBound, upperBound)
    cv.imwrite("maskR1.png", maskR1)
    
    # Creating R2 Mask ()
    imgRGB = normalized(img)
    lowerBound = np.array((0, 71, 92))
    # lowerBound = np.array((0, 28, 36))
    upperBound = np.array((255, 93, 119))
    # upperBound = np.array((1000, 363, 465))
    maskR2 = cv.inRange(imgRGB, lowerBound, upperBound)

    # maskR2[valid_range] = 255
    # maskR2[np.logical_not(valid_range)] = 0
    cv.imwrite("maskR2.png", maskR2)
    
    filteredImgHSV = cv.bitwise_and(imgHSV, imgHSV, mask=maskR1)
    cv.imwrite("HSV.png", filteredImgHSV)
    
    filteredImgRGB = cv.cvtColor(filteredImgHSV, cv.COLOR_HSV2BGR)
    cv.imwrite("convertedRGB.png", filteredImgRGB)
    
    filteredImgRGB = cv.bitwise_and(filteredImgRGB, filteredImgRGB, mask=maskR2)
    cv.imwrite("endImg.png", filteredImgRGB)
    
    return filteredImgRGB
    # normalized(img)

# def normalized(img):
    
#     # imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#     norm=np.zeros((img.shape[0],img.shape[1],3),np.float32)
#     norm_rgb=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
#     sum = np.zeros((img.shape[0],img.shape[1]), np.int32)
    
#     # Split the R, G, & B channels
#     rChannel, gChannel, bChannel = cv.split(img)
    
#     print("First Pixel")
#     pprint.pprint(rChannel[0][0])
#     pprint.pprint(gChannel[0][0])
#     pprint.pprint(bChannel[0][0])
    
#     print(rChannel.dtype)
    
#     # Getting the sum
#     print("Sum", int(rChannel[0][0]) + int(gChannel[0][0]) + int(bChannel[0][0]))
#     sumChannels = rChannel.astype(int) + gChannel.astype(int) + bChannel.astype(int)
#     print("Sum")
#     pprint.pprint(sumChannels)
#     rChannel = rChannel/sumChannels 
#     gChannel = gChannel/sumChannels 
#     bChannel = 1 - rChannel - gChannel
#     print("rChannel normalized")
#     pprint.pprint(rChannel)
    
    
#     # Scaling
#     rChannel *= 255
#     gChannel *= 255
#     bChannel *= 255
#     print("rCHannel scaled")
#     pprint.pprint(rChannel)
    
#     norm[:,:,0] = rChannel
#     norm[:,:,1] = gChannel
#     norm[:,:,2] = bChannel
#     cv.imwrite("normalizedUnscaled.png", norm)

#     norm_rgb=cv.convertScaleAbs(norm)
#     pprint.pprint(norm_rgb)
#     cv.imwrite("normalized.png", norm_rgb)
#     return norm_rgb

def normalized(img):
        # imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        norm=np.zeros((img.shape[0],img.shape[1],3),np.float32)
        norm_rgb=np.zeros((img.shape[0],img.shape[1],3),np.uint8)

        b=img[:,:,0]
        g=img[:,:,1]
        r=img[:,:,2]

        sum=b.astype(int) + g.astype(int) + r.astype(int)
        # pprint(sum)

        norm[:,:,0]=b/sum*255.0
        norm[:,:,1]=g/sum*255.0
        norm[:,:,2]=r/sum*255.0

        norm_rgb=cv.convertScaleAbs(norm)
        pprint.pprint(norm_rgb)
        cv.imwrite("normalized.png", norm_rgb)
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
while int(algorithmNo) not in [1, 2, 3, 4]:
    print("******************************************")
    print("\t\tAlgorithms:")
    print("******************************************")
    print("Peer et al. (RGB color space):\t\t", 1)
    print("Lin et al. (HSI color space):\t\t", 2)
    print("Chai & Ngan (YCbCr color space):\t", 3)
    print("Wang & Yuan (HSV & normalized RGB):\t", 4)
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

    newFileName = 'Results/chai-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Chai & Ngan Algorithm\n====================")
    processedImage = chaiAlgo(img)
else:
    newFileName = 'Results/wang-output-' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Wang & Yuan Algorithm\n====================")
    processedImage = wangAlgo(img)

# Ending timer
endTime = time.time()
timeTaken = round(endTime - startTime, 3)
print("Time taken: ", timeTaken, " seconds")

# Saving processed image
cv.imwrite(newFileName, processedImage)
