#!/usr/bin/python

import os, sys
import json
import numpy as np
import re



from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt

# ascent = misc.ascent()
# scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
#                    [-10+0j, 0+ 0j, +10 +0j],
#                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
# grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')

# fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
# ax_orig.imshow(ascent, cmap='gray')
# ax_orig.set_title('Original')
# ax_orig.set_axis_off()
# ax_mag.imshow(np.absolute(grad), cmap='gray')
# ax_mag.set_title('Gradient magnitude')
# ax_mag.set_axis_off()
# ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
# ax_ang.set_title('Gradient orientation')
# ax_ang.set_axis_off()
# fig.show()


#################################################################
###                                                           ###
###                Start Of Custom Functions                  ###
###                                                           ###
#################################################################

def binarised(B):
    B = np.array(B > 0)
    return B.astype(int)

def identifyAndCrop(x):
    #Assumptions: That x is a binary 2D array. So, containing integers only in the set {0,1}
    global template_zone_size
    #Fancy indexing
    template = x[:template_zone_size[0]-1,:template_zone_size[1]-1]
    #Check if we can crop the template to make the convolution better, both in computation and in how close it can get to the window edges.
    #sum columns and rows independently to see if any are empty (meaning we could discard that row or column from our template)
    col_sum = template.sum(axis=0)
    row_sum = template.sum(axis=1)
    #Get the bounds of our template (it may be smaller than the 3x3 grid size), by finding the columns and rows that sum to non-zero totals.
    col_crop = [idx for idx, n in enumerate(col_sum) if n!=0]
    row_crop = [idx for idx, n in enumerate(row_sum) if n!=0]
    print(row_sum)
    print("Row_crop = ", row_crop)
    print(col_sum)
    print("Col_crop = ", col_crop)
    #Crop the template to the size determined by the first and last non-zero totals in columns and rows.
    print(row_crop[0])
    print(row_crop[-1])
    print(col_crop[0])
    print(col_crop[-1])
    #Crop the template to the dimensions of the non-xero rows and columns calculated in the list comprehensions above.
    
    out = template[row_crop[0]:row_crop[-1]+1,col_crop[0]:col_crop[-1]+1]
    return out

def template_flip(T):
    #Flip the template in advance of the convolution.
    out = np.copy(T)
    #Flip about each axis.
    out = np.flip(out, 0)
    out = np.flip(out, 1)
    return out
        
#################################################################
###                                                           ###
###                 End Of Custom Functions                   ###
###                                                           ###
#################################################################



### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_63613498(x):
    #Assumptions: 
    #1. Background is always zero.
    #   Could use a histogram to identify the most common cell number (which would give 0 as a background)
    #   A problem with using a histogram is that the background may not necessarily be the most common cell value. 
    #   Also, we have not been told that background can change from test to test. Hence why I'm making the "background = 0" assumption.  
    #
    #2. The square at the top of the tile is fixed in position and size (ie 3x3) for all problems.
    #3. There is only one match. If there are multiple equivalent results, we just ignore all but the first "best" result encountered.
    #4. Template shapes can be assumed to occupy the full area inside the bounds on each axis. ie templates are always rectangular.
    #
    
    #Making global for debugging
    #global y
    #Define the template we're looking for.
    global template_zone_size 
    template_zone_size = (4,4)
    
    #As colour is not the same between template and the shape we're looking for, we will binarise the image so we can use convolution later to identify the best match to our template. 
    binary = binarised(x)
    
    template = identifyAndCrop(binary)
    
    #Pre-flip the template in advance of the convolution.
    #template_flipped = template_flip(template)
    template_flipped = np.copy(template)
    
    #"Zero-out" the template and boundary in the original image so our convolution doesnt run on it and generate a match from itself.
    binary[0:template_zone_size[0],0:template_zone_size[1]] = np.zeros(template_zone_size)
    #Convolve the template over the image to create a heatmap of the best matches.
    #y = signal.convolve2d(binary, template_flipped, boundary='symm', fillvalue=0, mode='same')
    y = signal.correlate2d(binary, template_flipped, boundary='symm', fillvalue=0, mode='same')
    
    
    #Find the image location with the max value from the convolution.
    L = np.unravel_index(np.argmax(y),np.shape(y))
    
    
    #Identify the colour / number value to assign to the pixels that make up the best match.
    #These always match the boundary of the 3x3 grid at the top left of the image.
    colour = x[template_zone_size[0]-1,template_zone_size[1]-1]
    print("Colour is: ",colour)
    #Replace the match in the original image with our template (including an updated colour to match the boundary).
    #start_row = L[0] - np.shape(template)[0]
    #start_col = L[1] - np.shape(template)[1]
    #x[start_row:start_row+template_zone_size[0],start_col:start_col+template_zone_size[1]] = 10
    x[L[0],L[1]] = 10
    #Plot result.
    print("binary: ", binary)
    print("Template: ", template)
    print("Template flipped: ", template_flipped)
    print("y: ", y)
    print("Location: ", L)
    print("Template shape: ", np.shape(template))
    fig, (ax_orig, ax_result) = plt.subplots(2, 1, figsize=(6, 15))
    ax_orig.imshow(x, cmap='hsv')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_result.imshow(y, cmap='gray')
    ax_result.set_title('Result')
    ax_result.set_axis_off()
    fig.show()
    return y



def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        try:
            directory = os.path.join("..", "data", "training")
            json_filename = os.path.join(directory, ID + ".json")
        except:
            print("Failed to open pre-set path. Using Mikes full path.")
            json_filename = "C:\Repos\prog_and_tools_for_ai\PTAI_Assignment_3\ARC\data\training"
            os.chdir(json_filename)
            json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

