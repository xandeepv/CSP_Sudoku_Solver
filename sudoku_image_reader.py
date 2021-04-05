'''
CS 5233 AI Team Project
Fall 2020
Sudoku Image reader is Computer vision utlity to read an 
image and identify the sudoku grid. This will also split
the sudoku grid in to 81 segments for predicting the intial
stage of the sudoku given by the image. 

Dependency:
The Digit Character trained model inputfile "SudoVision_RT.h5" should be
present in the same directory as the this file.

Usage: sudoku_image_reader.py <[/f | /file]:[filename]>
Example:
sudoku_image_reader.py /f sudo1.png

For importing as a module
from sudoku_image_reader import *

Note: If an invalid Sudoku image is provided, "No Sudoku grid found" will be displayed.  
'''

import cv2
import numpy as np
import pandas as pd
import operator
from imutils import contours
import imutils
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from os import path
import sys, time, os

# Reading model for Number recognition
# model_file = 'SudoVision_RT_mnist.h5'
model_file = 'SudoVision_RT.h5'
number_model = tf.keras.models.load_model(model_file)
print("load model file:",model_file)
#Parameters for Warping the image
margin = 10
case = 28 + 2*margin
perspective_size = 9*case


# the below fucntion gets the filename to create the intial state of the Sudoku puzzle
# Also get the image formats to display the solved sudoku result image back on the image
# 
def image_read_get_grid_val(filename):
    frame = cv2.imread(filename)
    p_frame = frame.copy()


    #Process the frame to find contour
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray=cv2.GaussianBlur(gray, (5, 5), 0)
    thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)


    #Get all the contours in the frame
    contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    maxArea = 0


    #Find the largest contour(Sudoku Grid)
    for c in contours_:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, 0.01*peri, True)
            if area>maxArea and len(polygon)==4:
                contour = polygon
                maxArea = area

    #Draw the contour and extract Sudoku Grid
    if contour is not None:
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
        points = np.vstack(contour).squeeze()
        points = sorted(points, key=operator.itemgetter(1))

        if points[0][0]<points[1][0]:
            if points[3][0]<points[2][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0]<points[2][0]:
                pts1 = np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1 = np.float32([points[1], points[0], points[2], points[3]])

        pts2 = np.float32([[0, 0], [perspective_size, 0], [0, perspective_size], [perspective_size, perspective_size]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_window =cv2.warpPerspective(p_frame, matrix, (perspective_size, perspective_size))
        result = perspective_window.copy()



        #Process the extracted Sudoku Grid
        p_window = cv2.cvtColor(perspective_window, cv2.COLOR_BGR2GRAY)
        p_window = cv2.GaussianBlur(p_window, (5, 5), 0)
        p_window = cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        p_window = cv2.morphologyEx(p_window, cv2.MORPH_CLOSE, vertical_kernel)
        lines = cv2.HoughLinesP(p_window, 1, np.pi/180, 120, minLineLength=40, maxLineGap=10)



        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #Invert the grid for digit recognition
        invert = 255 - p_window
        invert_window = invert.copy()




        invert_window = invert_window /255


        ## GET the n=81 parallel 28x28 images for prediction
        def get_all_81(xy,inv_win=invert_window):
            ymin = xy['ymin']
            ymax = xy['ymax']
            xmin = xy['xmin']
            xmax = xy['xmax']
            img = inv_win[ymin:ymax, xmin:xmax]
            x_pos = int((xmin + xmax)/ 2)-5
            y_pos = int((ymin + ymax)/ 2)+10
            img = cv2.resize(img,(28,28))
            img = img.reshape((1,28,28,1))
            pixel_sum = np.sum(img)
            if pixel_sum > 775.0:
                pred = 0
            else:
                pred = 100
                x_pos = 0
                y_pos = 0
            return [img,pred, pixel_sum, (x_pos, y_pos)]


        vec_df = pd.DataFrame(list(zip([10, 58, 106, 154, 202, 250, 298, 346, 394]*9, [38, 86, 134, 182, 230, 278, 326, 374, 422]*9, np.sort([10, 58, 106, 154, 202, 250, 298, 346, 394]*9), np.sort([38, 86, 134, 182, 230, 278, 326, 374, 422]*9))), columns =['xmin', 'xmax','ymin','ymax'])
        vec_df.head(3)
        vec_df[['img','pred','pixel_sum','x_y_pos']] = vec_df.apply(get_all_81, axis=1, result_type='expand')

        vec_df['model_pred'] = tf.keras.backend.argmax(number_model.predict(np.array(list(vec_df['img'])).reshape(81, 28, 28, 1)), axis=1).numpy()
        vec_df['final_pred'] = vec_df[['model_pred','pred']].apply(lambda x: int(x[0]) if x[1]==100 else int(x[1]),axis=1)
        print(np.asarray(vec_df['final_pred']).reshape(9,9))
        
    return ''.join(map(str, vec_df['final_pred'])), result, matrix, vec_df['x_y_pos']


def display_result_on_image(pred_vec, result, matrix, x_y_pos):
    for i in range(81):
        if x_y_pos[i] != (0,0):
            result = cv2.putText(result, str(pred_vec[i]), x_y_pos[i], cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", result)
    # frame = cv2.warpPerspective(result, matrix, (perspective_size, perspective_size), flags=cv2.WARP_INVERSE_MAP)
    # cv2.imshow("frame", frame)
    #cv2.imshow('P-Window', p_window)
    #cv2.imshow('Invert', invert)
    cv2.waitKey(2)&0xFF

    
    
    
    

# program main entry point
if __name__ == '__main__':
    error=False
    if len(sys.argv)>=2 and len(sys.argv)<=4:
        for i, arg in enumerate(sys.argv):
            if i==0:
                continue
            elif (str(arg).startswith('/f') or str(arg).startswith('/f')):
                file_name = str(arg).replace('/f:','').replace('/file:','')
                print(file_name)
                if not path.exists(file_name):
                    error = True
                    print('File do not exist')
                    break
            else:
                error = True        
                break
        
    
        if not error:
            start_time = time.perf_counter()
            print(image_read_get_grid_val(file_name))
            end_time = time.perf_counter()
            print(f'Solved in {end_time - start_time:0.4f} seconds.')
        else:
            print('Usage: sudoku_image_reader.py <[/f | /file]:[filename]>')
            
    else:
        print('Usage: sudoku_image_reader.py <[/f | /file]:[filename]>')
        
