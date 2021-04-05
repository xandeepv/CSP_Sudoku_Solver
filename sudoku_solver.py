'''
CS 5233 AI Team Project
Fall 2020
Sudoku problem solver which solves the Sudoku puzzle by defining the Sudoku Solver 
as a constraint satisfaction problem and using a model-based, goal-based agent 
that uses inference, heuristics, and constraint propagation strategies to find a solution.

Usage: sudoku_solver.py <board values> </g | /grid> <[/f | /file]:[filename]>
Example:
sudoku_solver.py 003020600900305001001806400008102900700000008006708200002609500800203009005010300 /g

Note: If an invalid Sudoku initial state is provided, "No solution found." will be displayed.  
'''
import sys, time, os
import numpy as np

# cross product of two lists
def cross_product(list1, list2):
    list3 = []
    for e1 in list1:
        for e2 in list2:
            list3.append(e1 + e2)
    return list3

# Sudoku board constructs
NUM_SUDOKU_SQUARES = 81
BOX_WIDTH = 3
letter_tuples = ('ABC','DEF','GHI')
number_tuples = ('123','456','789')
nums = ''.join(number_tuples)
columns = nums
rows = ''.join(letter_tuples)
squares = cross_product(rows, columns)
column_units = [cross_product(rows, col) for col in columns]
row_units = [cross_product(row, columns) for row in rows]
box_units = [cross_product(rboxes, cboxes) for rboxes in letter_tuples for cboxes in number_tuples]
unit_list = (column_units + row_units + box_units)
units = dict((square, [unit for unit in unit_list if square in unit]) for square in squares)
unit_peers = dict((square, set(sum(units[square],[]))-set([square])) for square in squares)

def parse_sudoku_board(sudoku_board):
    "Convert sudoku board to a dictionary of possible square numbers"
    try:
        values = dict((square, nums) for square in squares)
        for square, num in sudoku_board_values(sudoku_board).items():
            if num in nums and not assign(values, square, num):
                return False
    except Exception as e:
        print("An error occurred in parse_sudoku_board:", e)
    return values

def sudoku_board_values(sudoku_board):  
    "Convert sudoku board into a dictionary of square locations to numbers with zero's (0) for unfilled/blank squares"
    square_values = None
    try:        
        if isinstance(sudoku_board, str):
            chars = [char for char in sudoku_board if char in nums or char in '0']       
        elif isinstance(sudoku_board, np.ndarray):
            sudoku_board = sudoku_board.flatten()
            chars = ''.join(str(i) for i in sudoku_board.tolist())
        elif isinstance(sudoku_board, list):
            chars = ''.join(str(i) for i in sudoku_board)
        assert len(chars) == NUM_SUDOKU_SQUARES
        square_values = dict(zip(squares, chars))
    except Exception as e:
        print("An error occurred in sudoku_board_values:", e)
    return square_values

def assign(values, square, num):
    "remove all numbers but num from values and propagate. Return values otherwise return False"
    try: 
        not_num_values = values[square].replace(num, '')
        if all(constraint_propagation(values, square, num2) for num2 in not_num_values):
            return values
        else:
            return False
    except Exception as e:
        print("An error occurred in assign:", e)

def constraint_propagation(values, square, num):
    try:
        #display(values, grid=True)
        "Eliminate number from search space and and propagate"
        if num not in values[square]: # check if number is already removed from remaing values
            return values
        values[square] = values[square].replace(num,'') # otherwise remove from remaining values
        # if a square is reduced to one value number, then apply constraint propagation to it from it's unit peers
        if len(values[square]) == 0:
            return False
        elif len(values[square]) == 1:
            num2 = values[square]
            if not all(constraint_propagation(values, square2, num2) for square2 in unit_peers[square]):
                return False
        # if a unit is reduced to only one place for a number the assign it to square
        for unit in units[square]:
            num_places = [square for square in unit if num in values[square]]
            if len(num_places) == 0: # no places
                return False
            elif len(num_places) == 1:
                # number can only be in one place in a unit so assign it to that place
                if not assign(values, num_places[0], num):
                    return False
        return values
    except Exception as e:
        print("An error occurred in constraint_propagation:", e)

def display(values, grid=False):
    "Display these values as a 2-D Sudoku board."
    try:
        if grid:
            width = max(len(values[square]) for square in squares) + 1
            line = '+'.join(['-'*(width*BOX_WIDTH)]*BOX_WIDTH)
        for row in rows:
            for col in columns:
                if grid:
                    sq_val = values[row+col].center(width)
                else:
                    sq_val = values[row+col]
                grid_char=''
                if grid and col in '36':                
                    grid_char='|'
                if sq_val+grid_char != None:              
                    print(('').join(sq_val+grid_char), end='')
            print()
            if grid and row in 'CF': print(line)
        print()
    except Exception as e:
        print("An error occurred in display:", e)

def solve(sudoku_board): 
    "Solve Sudoku puzzle given initial values"
    solution = None
    try:
        solution = dfs(parse_sudoku_board(sudoku_board))
    except Exception as e:
        print("An error occurred in solve:", e)
    return solution

def minimum_remaining_values(values):
    "use minimum remaining values (MRV) heuristic to chose the square with the fewest legal values"
    try:        
        list = []
        for square in squares:
            if len(values[square]) > 1:
                list.append((len(values[square]), square))
        n, square = min(tuple(list))
    except Exception as e:
        print("An error occurred in MRV:", e)
    return square

def dfs(values):
    "depth first search (dfs)"
    try:        
        if values is False:
            return False
        # check if all squares are filled in with one value
        if all(len(values[square]) == 1 for square in squares): 
            return values
        # use MRV heuristic
        square = minimum_remaining_values(values)
        # perform a depth first search (dfs) of all possibilities under values[square] == num 
        # before we consider a different value for square
        search_results = []        
        for num in values[square]:
            remaining_values = assign(values.copy(), square, num)
            search_results.append(dfs(remaining_values))        
        return get_first_true_element(search_results)
    except Exception as e:
        print("An error occurred in dfs:", e)

# get the first true element value in the sequence
def get_first_true_element(e_sequence):
    try:
        for element in e_sequence:
            if element: return element
    except Exception as e:
        print("An error occurred in get_first_true_element:", e)
    return False

# read initial Sudoku board from file
def read_file(filename, sep='\n'):
    content = None
    data_dir = '.\\' 
    file_exists = True
    try:
        if os.path.exists(filename):
            fullFilePath = filename
        elif os.path.exists(os.path.join(data_dir, filename)):
            fullFilePath = os.path.join(data_dir, filename)
        else:
            file_exists = False
        if file_exists:
            f = open(fullFilePath, "r")
            content = f.read().strip().split(sep)
            f.close()
        else:
            print("File not found:", filename)  
    except Exception as e:
        print("An error occurred in read_file:", e)  
    return content

# program main entry point
if __name__ == '__main__':
    showGrid = False
    error = False
    if len(sys.argv)>=2 and len(sys.argv)<=4:        
        for i, arg in enumerate(sys.argv):
            if i==0:
                continue
            elif (str(arg)=='/g' or str(arg)=='grid'):
                showGrid = True
            elif (str(arg).startswith('/f') or str(arg).startswith('/f')):
                file_name = str(arg).replace('/f:','').replace('/file:','')
                sudoku_board = ''.join(read_file(file_name, sep='\n'))
            elif len(arg) == NUM_SUDOKU_SQUARES:
                sudoku_board = arg
            else:
                error = True        
                break

        if not error:            
            if len(sudoku_board) == NUM_SUDOKU_SQUARES:
                display(sudoku_board_values(sudoku_board), showGrid)
                start_time = time.perf_counter()
                values = solve(sudoku_board)
                end_time = time.perf_counter()
                if values: 
                    display(values, showGrid)
                    print(f'Solved in {end_time - start_time:0.4f} seconds.')
                else:
                    print('No solution found.')
            else:
                print(f'Invalid Sudoku input. String of {NUM_SUDOKU_SQUARES:d} consecutive integers expected.')
        else:
            print('Usage:: sudoku_solver.py <board values> </g | /grid> <[/f | /file]:[filename]>')
    else:
        print('Usage: sudoku_solver.py <board values> </g | /grid> <[/f | /file]:[filename]>')
