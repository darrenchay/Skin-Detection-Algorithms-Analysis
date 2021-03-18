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
originalImg = cv.imread(filename, -1)  # reading image 
img = np.float32(originalImg)/255

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

maskR1 = cv.inRange(imgLab, int(Lmean), 100)
result = cv.bitwise_and(img, img, mask=maskR1)

plt.subplot(1, 2, 1)
plt.imshow(maskR1, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()


# plt.subplot(121),plt.imshow(L),plt.title('L')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(A),plt.title('a')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(B),plt.title('b')
# plt.xticks([]), plt.yticks([])
# plt.show()
# cv.imshow("L_Channel",L) # For L Channel
# cv.imshow("A_Channel",A) # For A Channel (Here's what You need)
# cv.imshow("B_Channel",B) # For B Channel
# cv.waitKey(0)
# cv.destroyAllWindows()
imgRGB = cv.cvtColor(imgLab, cv.COLOR_LAB2RGB)
# plt.imshow(imgRGB)
# plt.show()
exit()