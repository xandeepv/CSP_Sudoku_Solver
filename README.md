# CSP_Sudoku_Solver
Constraint Satisfaction Problem based Sudoku Solution, to extract sudoku problems from images or video streams to get 

Project Name: SudoVision RT (RT for Real Time)


Character Recognition Model trained Files:
SudoVision_RT.h5

Jupyter Notebook which created the above model file:
SudoVision_RT_Model.ipynb

Python Implementation files/Source Code:
1. sudoku_image_reader.py - Reads an image or video frame with Sudoku Grid and extracts the indivdiual 81 elements of the 9x9 sudoku problem. Each element then uses the model file to determine the value in the original sudoku problem or blank is missing values.

2. sudoku_solver.py - Solves the extracted Sudoku grid as a CSP (Constraint Satisfaction Problem) or tells no possible solutions. 
3. sudovision_RT_main.py (Run this file to execute the combined solution to read image and solve 9x9 sudoku)

Tags: Machine Learning, Artificial Intelligence, Computer Vision, sudoku grid extraction from Images, 
Video stream analysis, Sudoku solution, Constraint Satisfaction Problem (CSP)

Copyright - The Owner needs to be tagged in any replication or clone of this program or part of the solutions.


For Students- Please do not use this code directly to solve your course projects. Try to put some effort and it is going to pay you well in future


