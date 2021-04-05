'''
CS 5233 AI Team Project
Fall 2020]

this is the main file to call the fucntionalities of the 
SudoVision RT project

Dependencies: (need to have the below files in the same directory)
1. sudoku_image_reader.py
2. sudoku_solver.py


Start the Project code by running on cmd:
$ python3 sudovision_RT_main.py


This program expects you to enter file name of Images of Sudoku 
Else enter "exit" or "stop"

''' 

import sys, time, os
from sudoku_image_reader import *
from sudoku_solver import *


# program main entry point
if __name__ == '__main__':
    looping = True
    while looping:
        file_name = input('Enter the Image file for the sudoku:')
        if (file_name == 'exit') or (file_name == 'stop'):
            looping = False
        else:
            if os.path.exists(file_name):
                start_time1 = time.perf_counter()
                sudoku_board, result, matrix, x_y_pos_vec = image_read_get_grid_val(file_name)
                end_time1 = time.perf_counter()
                print(f'Image processed in {end_time1 - start_time1:0.4f} seconds.')
                showGrid = False
                error = False
                if len(sudoku_board) == NUM_SUDOKU_SQUARES:
                    start_time = time.perf_counter()
                    values = solve(sudoku_board)
                    end_time = time.perf_counter()
                    display(sudoku_board_values(sudoku_board), showGrid)
                    if values:
                        pred_vec = list(map(int,values.values()))
                        display(values, showGrid)
                        print(f'Solved in {end_time - start_time:0.4f} seconds.')
                        display_result_on_image(pred_vec,result, matrix, x_y_pos_vec)
                    else:
                        print('No solution found.')
                    
                else:
                    print(f'Invalid Sudoku input. String of {NUM_SUDOKU_SQUARES:d} consecutive integers expected.')
            else:
                print('not a valid file name, try again')
        
        