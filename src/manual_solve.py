#!/usr/bin/python

import os, sys
import json
import numpy as np
import re



from scipy import signal
from scipy import misc
from scipy import ndimage
    
import matplotlib.pyplot as plt
from collections import defaultdict 

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
    #print(row_sum)
    #print("Row_crop = ", row_crop)
    #print(col_sum)
    #print("Col_crop = ", col_crop)
    #Crop the template to the size determined by the first and last non-zero totals in columns and rows.
    #print(row_crop[0])
    #print(row_crop[-1])
    #print(col_crop[0])
    #print(col_crop[-1])
    #Crop the template to the dimensions of the non-xero rows and columns calculated in the list comprehensions above.
    
    out = template[row_crop[0]:row_crop[-1]+1,col_crop[0]:col_crop[-1]+1]
    return template

def template_flip(T):
    #Flip the template in advance of the convolution.
    out = np.copy(T)
    #Flip about each axis.
    out = np.flip(out, 0)
    out = np.flip(out, 1)
    return out

def findCorners(x, class_value):
    #Top right (TR), Top left (TL), Bottom right (BR), and Bottom left (BL) corner templates that we'll look for.
    TR_template = np.array([[1,1,1],[0,0,1],[0,0,1]])
    TL_template = np.array([[1,1,1],[1,0,0],[1,0,0]])
    BR_template = np.array([[0,0,1],[0,0,1],[1,1,1]])
    BL_template = np.array([[1,0,0],[1,0,0],[1,1,1]])
    TR = signal.correlate2d(x, TR_template, boundary='fill', fillvalue=0, mode='same')
    TL = signal.correlate2d(x, TL_template, boundary='fill', fillvalue=0, mode='same')
    BR = signal.correlate2d(x, BR_template, boundary='fill', fillvalue=0, mode='same')
    BL = signal.correlate2d(x, BL_template, boundary='fill', fillvalue=0, mode='same')
    
    #Threshold each image so we only see the perfect matches for the template used.
    #Set the "ideal" score that we should get from a correlation of a template with it's match.
    
    
    #Make a dict database to hold the various corner-type templates and resulting corner heatmaps (results of correlations) etc.
    DB = dict()
    DB["TL"] = {"template": TL_template, "heatmap": TL}
    DB["TR"] = {"template": TR_template, "heatmap": TR}
    DB["BL"] = {"template": BL_template, "heatmap": BL}
    DB["BR"] = {"template": BR_template, "heatmap": BR}
    
    #For each corner type we're looking for:
    for idx, n in enumerate(DB):
        print(n, ":")
        T = DB[n]["template"]
        #Identify where the correlation peaks were, indicating corners.  Disregard any arrays that did not get an exact match as a corner.
        DB[n]["corners"] = [c for idx, c in enumerate(np.where(DB[n]["heatmap"] == np.sum(T))) if len(c) > 0]
        ########################Reformat corners into a list of [R,C] lists, rather than independent lists for R and C separately.
        print(f"Max score: {np.sum(T)}")
        print(DB[n]["heatmap"])
        print("Corners found: ", len(DB[n]["corners"]))
        print("Corners shape: ", np.shape(DB[n]["corners"]))
        print(DB[n]["corners"])
    
    #Use Lambda function to reformat corners so they are in [R1,C1], [R2,C2] etc format, rather than "[R1, R2,... RN], [C1, C2,... CN]" format.
    for idx, n in enumerate(DB):
        #Manipulate the format of corners to be in [r,c] pairs.
        f = lambda row,arr: [arr[0][row],arr[1][row]]
        reformatted_corners = [f(element,DB[n]["corners"]) for element in range(np.shape(DB[n]["corners"])[0])]
        DB[n]["corners"] = reformatted_corners

    #Correctly offset corners based on their template. 
    #Eg a bottom-right corner gets picked up at an offset of [+1,+1] because...
    #... the correlation reports results for the template, without taking...
    #... into account where the corner is within that template.        
    for idx, n in enumerate(DB):
        if n == "TL":
            f = lambda rc_coord: [rc_coord[0]-1, rc_coord[1]-1]
            corrected_corners = [f(coords) for coords in DB[n]["corners"]]
            DB[n]["corners"] = corrected_corners    
        if n == "TR":
            f = lambda rc_coord: [rc_coord[0]-1, rc_coord[1]+1]
            corrected_corners = [f(coords) for coords in DB[n]["corners"]]
            DB[n]["corners"] = corrected_corners    
        if n == "BL":
            f = lambda rc_coord: [rc_coord[0]+1, rc_coord[1]-1]
            corrected_corners = [f(coords) for coords in DB[n]["corners"]]
            DB[n]["corners"] = corrected_corners    
        if n == "BR":
            f = lambda rc_coord: [rc_coord[0]+1, rc_coord[1]+1]
            corrected_corners = [f(coords) for coords in DB[n]["corners"]]
            DB[n]["corners"] = corrected_corners
        print(n, ": ", DB[n]["corners"])
        
        
     
    fig, tile = plt.subplots(2, 2)
    fig.suptitle(f"Corner Scores Detected for Class: {class_value}")
    #Set vmax so we see only the perfect matches as the brightest cells.
    tile[0,0].imshow(TL, cmap='gray', vmax = np.sum(DB["TL"]["template"]))
    tile[0,0].set_title('Top Left')
    tile[0,1].imshow(TR, cmap='gray', vmax = np.sum(DB["TR"]["template"]))
    tile[0,1].set_title('Top Right')
    tile[1,0].imshow(BL, cmap='gray', vmax = np.sum(DB["BL"]["template"]))
    tile[1,0].set_title('Bottom Left')
    tile[1,1].imshow(BR, cmap='gray', vmax = np.sum(DB["BR"]["template"]))
    tile[1,1].set_title('Bottom Right')
    fig.show()
    
    return DB


    
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

def solve_fcb5c309(x):
    #Assumptions: 
    #1. The background is the most frequent colour in the grid.
    #2. The enclosing rectangles are all the same colour, and have only 4 sides. (ie they must be rectangles! Not just any polygon. )
    #3. The enclosing rectangles will not overlap.
    
    print("x is: ")
    print(x)
    #Make a histogram of all the tile values so we can start identifying how many different classes there are in the grid.
    #could use a dict like we used in lectures, but this is one line of code! :D
    hist = ndimage.measurements.histogram(x, 0, np.max(x), np.max(x)+1)
    #Zero-out the background in the histogram, because we don't want to analyse the background.
    hist[0] = 0
    #Create a list to gather all the data relating to each colour type (aka class) in the original image.
    global class_data
    class_data = dict()
    #Segment out each class, so we can analyse each colour of cell independently. 
    for idx, n in enumerate(hist):
        #If the frequency of occurrence of that class is zero, skip it.  Only process colours that exist in the original image. 
        if n !=0:
            #binarising and only propagating one specific cell value / class at a time.
            B = np.array(x == idx).astype(int)
            print(f"Finding values = {idx}")
            class_data[idx] = findCorners(B, idx)
            
    #Determine which class has the most corners.
    global corner_count
    corner_count = defaultdict(int)
    #Define which class has the most corners:
    global most_corners
    global most_corners_class_id
    most_corners = 0
    most_corners_class_id = 0
    for n in class_data:
        for corner_type in ["TL", "TR", "BL", "BR"]:
            print(n, " : ", class_data[n][corner_type]["corners"])
            print("#corners: ", len(class_data[n][corner_type]["corners"]))
            print(len(class_data[n][corner_type]["corners"]))
            print(n)
            corner_count[n] += len(class_data[n][corner_type]["corners"])
            if corner_count[n] > most_corners:
                most_corners = corner_count[n]
                most_corners_class_id = n

    #Now we have the class containing the most corners stored in "most_corners" with class id "most_corners_class_id".
    #Now lets find each of the four corners of each rectangle until they are all found.
    #class_data[most_corners_class_id]
    

    
    #B = binarised(x)
    
    #Segment out each class and check how rectangular they are.
    #Use correlation to find each of the 4 right angles in a rectangle. Then use an FSM to extract each rectangle.
    #findCorners(B)
    
    #y = binarised(x)
    
    
    #print(x)

    fig, (ax_orig, ax_result) = plt.subplots(1, 2, figsize=(15,6))
    fig.suptitle("Original (L) vs Result (R)")
    ax_orig.imshow(x, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_result.imshow(y, cmap='gray')
    ax_result.set_title('Result')
    ax_result.set_axis_off()
    fig.show()
    
    return x

# def solve_63613498(x):
#     #Assumptions: 
#     #1. Background is always zero.
#     #   Could use a histogram to identify the most common cell number (which would give 0 as a background)
#     #   A problem with using a histogram is that the background may not necessarily be the most common cell value. 
#     #   Also, we have not been told that background can change from test to test. Hence why I'm making the "background = 0" assumption.  
#     #
#     #2. The square at the top of the tile is fixed in position and size (ie 3x3) for all problems.
#     #3. There is only one match. If there are multiple equivalent results, we just ignore all but the first "best" result encountered.
#     #4. Template shapes can be assumed to occupy the full area inside the bounds on each axis. ie templates are always rectangular.
#     #
    
#     #Making global for debugging
#     #global y
#     #Define the template we're looking for.
#     global template_zone_size 
#     template_zone_size = (4,4)
    
#     #As colour is not the same between template and the shape we're looking for, we will binarise the image so we can use convolution later to identify the best match to our template. 
#     binary = binarised(x)
    
#     template = identifyAndCrop(binary)
    
#     #Pre-flip the template in advance of the convolution.
#     #template_flipped = template_flip(template)
#     template_flipped = np.copy(template)
    
#     #"Zero-out" the template and boundary in the original image so our convolution doesnt run on it and generate a match from itself.
#     binary[0:template_zone_size[0],0:template_zone_size[1]] = np.zeros(template_zone_size)
#     #Convolve the template over the image to create a heatmap of the best matches.
#     #y = signal.convolve2d(binary, template_flipped, boundary='symm', fillvalue=0, mode='same')
#     y = signal.correlate2d(binary, template_flipped, boundary='symm', fillvalue=0, mode='same')
    
    
#     #Find the image location with the max value from the convolution.
#     L = np.unravel_index(np.argmax(y),np.shape(y))
    
    
#     #Identify the colour / number value to assign to the pixels that make up the best match.
#     #These always match the boundary of the 3x3 grid at the top left of the image.
#     colour = x[template_zone_size[0]-1,template_zone_size[1]-1]
#     #print("Colour is: ",colour)
#     #Replace the match in the original image with our template (including an updated colour to match the boundary).
#     #start_row = L[0] - np.shape(template)[0]
#     #start_col = L[1] - np.shape(template)[1]
#     #x[start_row:start_row+template_zone_size[0],start_col:start_col+template_zone_size[1]] = 10
#     y = np.copy(x)
#     y[L[0]-1:L[0]+2,L[1]-1:L[1]+2] = np.multiply(binary[L[0]-1:L[0]+2,L[1]-1:L[1]+2],colour)
#     #Plot result.
#     #print("binary: ", binary)
#     #print("Template: ", template)
#     #print("Template flipped: ", template_flipped)
#     #print("y: ", y)
#     #print("Location: ", L)
#     #print("Template shape: ", np.shape(template))
#     fig, (ax_orig, ax_result) = plt.subplots(2, 1, figsize=(6, 15))
#     ax_orig.imshow(x, cmap='hsv')
#     ax_orig.set_title('Original')
#     ax_orig.set_axis_off()
#     ax_result.imshow(y, cmap='gray')
#     ax_result.set_title('Result')
#     ax_result.set_axis_off()
#     fig.show()
#     return y


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
        json_filename = ""
        try:
            directory = os.path.join("..", "data", "training")
            json_filename = os.path.join(directory, ID + ".json")
            data = read_ARC_JSON(json_filename)
        except:
            print("Failed to open pre-set path. Using Mikes full path as a fall-back.")
            directory = "C:\\Repos\\prog_and_tools_for_ai\\PTAI_Assignment_3\\ARC\\data\\training"
            os.chdir(directory)
            json_filename = os.path.join(directory, ID + ".json")
            print(json_filename)
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

