# Image Processing Libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# Utility Libraries
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# import pprint
import time

# Reading the file
Tk().withdraw()
filename = askopenfilename()
print("filename: ", filename)
img = cv.imread(filename, -1)  # reading image 
# img = np.float32(img)/255

# Convert from BGR to Lab format
imgLab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# Extracting L, a and b components
L,A,B = cv.split(imgLab)
Lmean = np.mean(L)
Amean = np.mean(A)
Bmean = np.mean(B)
# plt.imshow(L)
# plt.show()
print(np.mean(L))
print(np.mean(A))
print(np.mean(B))

print(np.amax(L))
print(np.amax(A))
print(np.amax(B))
# Extracting R1 Mask
lowerBound = np.array([int(Lmean), int(np.min(A)), int(np.min(B))])
upperBound = np.array([int(np.max(L)), int(np.max(A)), int(np.max(B))])
maskR1 = cv.inRange(imgLab, lowerBound, upperBound)
result1 = cv.bitwise_and(imgLab, imgLab, mask=maskR1)
# cv.imshow("Mask R1 Result",result)
# plt.subplot(1, 2, 1)
# plt.imshow(maskR1, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result1)
# plt.show()

# Extracting R2 Mask
lowerBound = np.array([int(np.min(L)), int(Amean), int(np.min(B))])
upperBound = np.array([int(np.max(L)), int(np.max(A)), int(np.max(B))])
maskR2 = cv.inRange(imgLab, lowerBound, upperBound)
result2 = cv.bitwise_and(imgLab, imgLab, mask=maskR2)
# cv.imshow("Mask R1 Result",result)
# plt.subplot(1, 2, 1)
# plt.imshow(maskR2, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result2)
# plt.show()

# Extracting R3 Mask
lowerBound = np.array([int(np.min(L)), int(np.min(A)), int(Bmean)])
upperBound = np.array([int(np.max(L)), int(np.max(A)), int(np.max(B))])
maskR3 = cv.inRange(imgLab, lowerBound, upperBound)
result3 = cv.bitwise_and(imgLab, imgLab, mask=maskR3)
# cv.imshow("Mask R1 Result",result)
plt.subplot(1, 3, 1)
plt.imshow(maskR1, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(maskR2, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(maskR3, cmap="gray")
plt.show()



imgRGB = cv.cvtColor(imgLab, cv.COLOR_LAB2RGB)
# plt.imshow(imgRGB)
# plt.show()