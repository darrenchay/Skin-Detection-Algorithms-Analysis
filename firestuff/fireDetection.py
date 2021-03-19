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

# Reading the file
Tk().withdraw()
filename = askopenfilename()
print("filename: ", filename)
img = cv.imread(filename, -1)  # reading image 
# img = np.float32(img)/255
pprint.pprint(img.shape)
# Convert from BGR to Lab format
imgLab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# Extracting L, a and b components
L,A,B = cv.split(imgLab)
Lmean = np.mean(L)
Amean = np.mean(A)
Bmean = np.mean(B)
pprint.pprint(A)
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

# Extracting R2 Mask
lowerBound = np.array([int(np.min(L)), int(Amean), int(np.min(B))])
upperBound = np.array([int(np.max(L)), int(np.max(A)), int(np.max(B))])
maskR2 = cv.inRange(imgLab, lowerBound, upperBound)
result2 = cv.bitwise_and(imgLab, imgLab, mask=maskR2)

# Extracting R3 Mask
lowerBound = np.array([int(np.min(L)), int(np.min(A)), int(Bmean)])
upperBound = np.array([int(np.max(L)), int(np.max(A)), int(np.max(B))])
maskR3 = cv.inRange(imgLab, lowerBound, upperBound)
pprint.pprint(maskR3)
result3 = cv.bitwise_and(imgLab, imgLab, mask=maskR3)

# Extracting R4 Mask
# pprint.pprint(img)
maskR4 = np.logical_and(A>B, A<np.max(A))
# img[maskR4] = 
result4 = cv.bitwise_and(imgLab, imgLab, mask=maskR4)
pprint.pprint(maskR4)

# R = [(30,70),(0,100),(0,100)]
# red_range = np.logical_and(R[0][0] < img[:,:,0], img[:,:,0] < R[0][1])
# green_range = np.logical_and(R[1][0] < img[:,:,0], img[:,:,0] < R[1][1])
# blue_range = np.logical_and(R[2][0] < img[:,:,0], img[:,:,0] < R[2][1])
# pprint.pprint(blue_range)
# valid_range = np.logical_and(red_range, green_range, blue_range)
# pprint.pprint(valid_range[0])
# print(valid_range[0].length())
# img[valid_range] = 100
# img[np.logical_not(valid_range)] = 0

# plt.imshow(img)
# plt.show()



plt.subplot(1, 3, 1)
plt.imshow(maskR1, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(maskR2, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(result4, cmap="gray")
plt.show()



imgRGB = cv.cvtColor(imgLab, cv.COLOR_LAB2RGB)
# plt.imshow(imgRGB)
# plt.show()