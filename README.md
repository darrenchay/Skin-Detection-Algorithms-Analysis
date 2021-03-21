# Content
## skinAlgorithms.py
This program prompts the user to select an image that they would like to process. Then, they will be able to choose between 3 skin detection algorithms:
- Peer et al. Algorithm using the RGB colour space
- Chai et al. Algorithm using the YCbCr colour space
- Wang & Yuan Algorithm using the HSV and normalized RGB colour space 
<br />
After choosing the algorithm, the image is processed and saved in the Results folder in the format 'algorithm-name-output-image-name.jpg'

## skinDetectionMetric.py
This program prompts the user to choose a ground truth image which will be used to compare with the processed image. <br />
Then, the user will be prompted to choose the processed image and the percentage of true and false positive will be calculated using the ground truth image as basis 
<br />
<b>True Positive </b>= (no. skin pixels in processed image) / (no. skin pixels in ground truth image) * 100
<br />
<b>False Positive </b>= (no. pixels which are not skin in processed image) / (no. of not skin pixels in ground truth image) * 100

## Results Folder
This folder contains the ground truth images as well as any processed image that is produced after running skinAlgorithms.py

# Running the programs

## Pre Conditions
- Make sure that there is a Results folder in the same folder that you are running the algorithm
- Make sure that you have python and the following libraries are installed first (using pip install):
    - numpy
    - scipy
    - opencv
    - matplotlib
    - tkinter

## Executing a skin detection algorithm
To run the program, type in 'python skinAlgorithms.py'. You will be prompted to select the image you want to run the algorithm of your choice and then input the number of the algorithm you want to choose. The result will then be stored in the Results folder after execution and the time taken to execute the algorithm will be printed out

## Executing the skin Detection Metric calculator
To execute the metric calculator, run the following command: 'python skinDetectionMetric.py', select the ground truth image of the image that you want to analyze and then choose the processed image that you obtained from running 'skinAlgorithms.py'. The value of the percentage true and false positive will then be calculated and printed out in the terminal in percentage format (to 3 decimal places) 

