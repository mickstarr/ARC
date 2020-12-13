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


### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.





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
    #col_sum = template.sum(axis=0)
    #row_sum = template.sum(axis=1)
    #Get the bounds of our template (it may be smaller than the 3x3 grid size), by finding the columns and rows that sum to non-zero totals.
    #col_crop = [idx for idx, n in enumerate(col_sum) if n!=0]
    #row_crop = [idx for idx, n in enumerate(row_sum) if n!=0]
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
    #out = template[row_crop[0]:row_crop[-1]+1,col_crop[0]:col_crop[-1]+1]
    
    return template

def template_flip(T):
    #For convolution before I remembered correlation! :D
    #Flip the template in advance of the convolution.
    out = np.copy(T)
    #Flip about each axis.
    out = np.flip(out, 0)
    out = np.flip(out, 1)
    return out

def findCorners(x, class_value):
    #Params: x is a binary image containing only cells that matched the colour "class_value"
    
    #Top right (TR), Top left (TL), Bottom right (BR), and Bottom left (BL) corner templates that we'll look for.
    TR_template = np.array([[1,1,1],[0,0,1],[0,0,1]])
    TL_template = np.array([[1,1,1],[1,0,0],[1,0,0]])
    BR_template = np.array([[0,0,1],[0,0,1],[1,1,1]])
    BL_template = np.array([[1,0,0],[1,0,0],[1,1,1]])
   
    #Correlation will be used to find the best matches between a given tempalte (right angle) anda given binary image.
    #Each of the below outputs are a "heatmap" image showing the correlation of each pixel to thew template corner. 
    TR = signal.correlate2d(x, TR_template, boundary='fill', fillvalue=0, mode='same')
    TL = signal.correlate2d(x, TL_template, boundary='fill', fillvalue=0, mode='same')
    BR = signal.correlate2d(x, BR_template, boundary='fill', fillvalue=0, mode='same')
    BL = signal.correlate2d(x, BL_template, boundary='fill', fillvalue=0, mode='same')
    
    #Make a dict database to hold the various corner-type templates and resulting corner heatmaps (results of correlations) etc.
    DB = dict()
    DB["TL"] = {"template": TL_template, "heatmap": TL}
    DB["TR"] = {"template": TR_template, "heatmap": TR}
    DB["BL"] = {"template": BL_template, "heatmap": BL}
    DB["BR"] = {"template": BR_template, "heatmap": BR}
    
    #For each corner type we're looking for, extract the best matches that were found from each heatmap:
    for idx, n in enumerate(DB):
        #the best correlation will be equal to the sum of all pixels in our binary corner template.  In this case, 5.
        T = DB[n]["template"]
        #Identify where the correlation peaks were, indicating corners.  Disregard any arrays that did not get an exact match as a corner.
        DB[n]["corners"] = [c for idx, c in enumerate(np.where(DB[n]["heatmap"] == np.sum(T))) if len(c) > 0]

        #Some debug prints:
        #print(f"Max score: {np.sum(T)}")
        #print(DB[n]["heatmap"])
        #print("Corners found: ", len(DB[n]["corners"]))
        #print("Corners shape: ", np.shape(DB[n]["corners"]))
        #print(DB[n]["corners"])
    
    #Use Lambda function to reformat corners so they are in [R1,C1], [R2,C2] etc format, rather than two lists in "[R1, R2,... RN], [C1, C2,... CN]" format.
    for idx, n in enumerate(DB):
        #Manipulate the format of corners to be in [r,c] pairs.
        f = lambda row,arr: [arr[0][row],arr[1][row]]
        reformatted_corners = [f(element,DB[n]["corners"]) for element in range(np.shape(DB[n]["corners"])[0])]
        DB[n]["corners"] = reformatted_corners

    #Correctly offset corners based on their template. 
    #Eg a bottom-right corner gets picked up at an offset of [+1,+1] because...
    #... the correlation reports results for the template, without taking...
    #... into account where the corner is within that template. We need to correct for this.       
    for idx, n in enumerate(DB):
        if n == "TL":
            #Lambda function to do the offsetting.
            f = lambda rc_coord: [rc_coord[0]-1, rc_coord[1]-1]
            #Iterate through all corners and apply the lambda function.
            corrected_corners = [f(coords) for coords in DB[n]["corners"]]
            #Update the corners in the database with the corrected ones. 
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
        
        
    #Print some debug images:
    # fig, tile = plt.subplots(2, 2)
    # fig.suptitle(f"Corner Scores Detected for Class: {class_value}")
    # #Set vmax so we see only the perfect matches as the brightest cells.
    # tile[0,0].imshow(TL, cmap='gray', vmax = np.sum(DB["TL"]["template"]))
    # tile[0,0].set_title('Top Left')
    # tile[0,1].imshow(TR, cmap='gray', vmax = np.sum(DB["TR"]["template"]))
    # tile[0,1].set_title('Top Right')
    # tile[1,0].imshow(BL, cmap='gray', vmax = np.sum(DB["BL"]["template"]))
    # tile[1,0].set_title('Bottom Left')
    # tile[1,1].imshow(BR, cmap='gray', vmax = np.sum(DB["BR"]["template"]))
    # tile[1,1].set_title('Bottom Right')
    # fig.show()
    
    return DB

def findNextCorner(current_state, current_corner, DB):
    #Should be part of the state machine I keep talking about! :P
    #If current state is the top-left corner, look for the next corner in a clockwise direction... ie the top-right corner.
    if current_state == "TL":
        candidate = current_corner
        #Given current state "top-left" we wish to move to state "top-right"
        #Before we do that though, we need to find the coordinate of the top-right corner we wish to move to.
        while candidate not in DB["TR"]["corners"]:
            candidate = [candidate[0], candidate[1]+1]
        #Now we have found the next corner in the TL, TR, BL, BR sequence, so we return this as the next valid corner.
        #Outside this function, we are (conceptually at least) moving to the next state in our (conceptual, due to time pressure) FSM. 
        return candidate
    
    if current_state == "TR":
        candidate = current_corner
        while candidate not in DB["BR"]["corners"]:
            candidate = [candidate[0]+1, candidate[1]]
        #Now we have found the next corner in the TL, TR, BL, BR sequence, so we return this as the next valid corner.
        return candidate
    
    if current_state == "BR":
        candidate = current_corner
        while candidate not in DB["BL"]["corners"]:
            candidate = [candidate[0], candidate[1]-1]
        #Now we have found the next corner in the TL, TR, BL, BR sequence, so we return this as the next valid corner.
        return candidate
    
    
def getBestReplacement(r,c,y):
    #Build a dataset of the pixel positions that should be the same colour as this [r,c] pixel.
    #Take the most common colour (the mode of our histogram of "non-corrupted" pixels) and replace the value of [r,c] with that.
    #Be careful to not use any yellow pixels, as they are the corrupted ones we wish to replace.
    
    #Data stores the coordinates of the pixels that should be similar to the current one we are evaluating.
    #These are deterministic due to the mirroring of the pattern about the x and y axes.
    data = []
    #Delta stores the offset between the center of the image and the current pixel under evaluation.
    delta = [0,0]
    
    #For each quadrant in the image, calculate the delta between the center of the image and the pixel under evaluation.
    #We will use this delta later to the pixels whose pixel valuesshould be equal. Then we can spot the most likely value and replace corrupted pixels with that value. 
    if (r <= 7) & (c <= 7):
        #Top-Left
        delta = np.abs(np.array([7,7]) - np.array([r,c]))        
    elif (r <= 7) & (c > 7):
        #Top-Right
        delta = np.abs(np.array([7,8]) - np.array([r,c]))
    elif (r > 7) & (c <= 7):
        #Bottom-Left
        delta = np.abs(np.array([8,7]) - np.array([r,c]))
    elif (r > 7) & (c > 7):
        #Bottom-Right
        delta = np.abs(np.array([8,8]) - np.array([r,c]))

    #For the given input coordinate [r,c], calculate the actual coordinate of its mirrored pixels in the image.           
    TL =  np.array([7,7]) + np.multiply(delta,[-1,-1])
    TR =  np.array([7,8]) + np.multiply(delta,[-1,+1])
    BL =  np.array([8,7]) + np.multiply(delta,[+1,-1])
    BR =  np.array([8,8]) + np.multiply(delta,[+1,+1])
    
    #Gather these pixels coordinates, which we will then use to build a histogram of the pixel values at these coordinates.
    data.append(TL)
    data.append(TR)
    data.append(BL)
    data.append(BR)
    
    #For each pixel coordinate gather so far, build a dataset of these pixel values.
    #Ignore any pixel values == 4, as 4 is the corrupted colour.
    colours = []
    for n in data:
        #Yellows are always = 4. They are the incorrect pixels, so we ignore themm when calculating statistics.
        if y[n[0],n[1]] != 4 :
            #Gather the pixel values (that don't have value 4) into a list.
            colours.append(y[n[0], n[1]])
    
    #Generate a histogram of these pixel values, and take the mode of this histogram.  This is the "most probable" value for our current pixel.
    hist = ndimage.measurements.histogram(colours, 0, np.max(colours), np.max(colours)+1)
    out = np.argmax(hist)
    
    return out
            
        
#################################################################
###                                                           ###
###                 End Of Custom Functions                   ###
###                                                           ###
#################################################################


def solve_fcb5c309(x):
    #Assumptions: 
    #1. The background is the most frequent colour in the grid.
    #2. The enclosing rectangles are all the same colour, and have only 4 sides. (ie they must be rectangles! Not just any polygon. )
    #3. The enclosing rectangles will not overlap.
    global y
    
    #Make a histogram of all the tile values so we can start identifying how many different classes there are in the grid.
    #could use a dict like we used in lectures, but this is one line of code, so I'll take the shortcut! :D
    hist = ndimage.measurements.histogram(x, 0, np.max(x), np.max(x)+1)
    #Zero-out the background in the histogram, because we don't want to analyse the background when looking for peaks etc in the histogram.
    hist[0] = 0
    #Create a dict to gather all the data relating to each colour type (aka class) in the original image.
    global class_data
    class_data = dict()
    #Segment out each class, so we can analyse each colour of cell independently. 
    for idx, n in enumerate(hist):
        #If the frequency of occurrence of that class is zero, skip it.  Only process colours that exist in the original image. 
        if n !=0:
            #binarising and only propagating one specific cell value / class at a time.
            B = np.array(x == idx).astype(int)
            #Find the location of the corners that exists for cells of the current colour.
            #Returns an updated DB with the corners included.
            class_data[idx] = findCorners(B, idx)
            
    #Determine which class has the most corners.
    #using globals for lots fo things so I can debug easier. Should probably consider learning about the debugger!
    global corner_count
    #Use a default dict to accumulate counts of corners. This will save me having to initialise the count at 0 for each corner type.
    corner_count = defaultdict(int)
    #Define which class has the most corners, and what that count is:
    global most_corners
    global most_corners_class_id
    most_corners = 0
    most_corners_class_id = 0
    #For each class of clours in the original image, count all the coners that exist, and identify the colour with the most corners.
    #the colour with the most corners is likely the rectangle colour.
    for n in class_data:
        #Iterate through the corner types.
        for corner_type in ["TL", "TR", "BL", "BR"]:
            #Accumulate the corner counts for each type of corner in that colour class.
            corner_count[n] += len(class_data[n][corner_type]["corners"])
            #Keep track of the colour class with the most corners.
            if corner_count[n] > most_corners:
                most_corners = corner_count[n]
                most_corners_class_id = n

    #Now we have the class containing the most corners stored in "most_corners" with class id "most_corners_class_id".
    #Now lets find each of the four corners of each rectangle until all corners are assigned to a rectangle.
    #Take any top-left corner and go from there, clockwise around the rectangle.  
    #####If I had more time I'd implement this as a state machine, but I don't have time.
    #Identify how many rectangles we have based on the total number of corners and the assumption that the don't overlap at all.
    num_rectangles = np.int(np.divide(most_corners,4))
    corners_already_used = []
    global rectangles
    rectangles = []
    #ID of the corner in a given set group of (for example top-right) corners
    idx = 0
    #For each expected rectangle, move clockwise trying to associate the next corner to the current corner.
    #Eg, if we're at top-left, we look for top right along the same row. becaus erectangles don't overlap, it should be the next corner on that row. 
    for n in range(num_rectangles):
        curr_rectangle = dict()
        #Start at top left corner and go from there.
        #Arbitrarily pick the first top left corner available.
        current_corner = class_data[most_corners_class_id]["TL"]["corners"][0]
        #Grab a top-left corner that hasn't yet been used. this loop is required for subsequent rectangles, so we don't keep detecting the same rectangle.
        while (current_corner in corners_already_used):
            current_corner = class_data[most_corners_class_id]["TL"]["corners"][idx]
            #Iterate through corners until we find a previously unused one, then break out of the while loop.
            idx += 1
            
        #Now that we're going to use this corner, we need to add it to the list so we know not to use it again in another later rectangle.
        corners_already_used.append(current_corner)
        #Define our first starting point and move clockwise, building up our rectangle, corner by corner.
        curr_rectangle["TL"] = current_corner
        #Current state is "Top Left" (TL) and the coordinate of our current corner is curr_rectangle["TL"]. This informs the function about the SISA mechanism.
        #Look for Top right corneres when given a top left corner, and so on, clockwise aaround the rectangle. 
        #Class_data is included to inform the function what coloured cells we're interested in.
        curr_rectangle["TR"] = findNextCorner("TL", curr_rectangle["TL"], class_data[most_corners_class_id])
        curr_rectangle["BR"] = findNextCorner("TR", curr_rectangle["TR"], class_data[most_corners_class_id])
        curr_rectangle["BL"] = findNextCorner("BR", curr_rectangle["BR"], class_data[most_corners_class_id])
        #When we've got 4 corners for our rectangle, append it to our list of finished rectangles.
        rectangles.append(curr_rectangle)
        
    #Now that we have detected all rectangles, and have them stored nicely... 
    #Find how many cells are inside each rectangle (this will determins which rectangle to output):
    #Binarise the interior of the rectangle so we can easily count the number of coloured cells in there. 
    B = binarised(x)
    global counts
    counts = []
    global internal_val
    internal_val = []
    #For each rectangle, count the number of cells inside it, and also log the colour (internal_val) of the cells in the rectangle.
    for n in rectangles:
        counts.append(np.sum(B[n["TL"][0]+1:n["BR"][0],n["TL"][1]+1:n["BR"][1]]))
        #Look back to the original image to detect the colour of the cells inside the rectangle.
        internal_val.append(np.max(x[n["TL"][0]+1:n["BR"][0],n["TL"][1]+1:n["BR"][1]]))
    #The desired rectangle is the one with the most cells inside it.
    best_rectangle_id = np.argmax(counts)
    #Assign the dict of corners of the best rectangle to the variable "b", chosen for it's short variable name.
    b = rectangles[best_rectangle_id]
    
    #Crop out the rectangle we want to return as our answer.
    y = B[b["TL"][0]:b["BR"][0]+1,b["TL"][1]:b["BR"][1]+1]
    #Scale the colours to match the original rectangle:
    y = np.multiply(y,internal_val[best_rectangle_id])
    
    
    ##Debug plots
    # fig, (ax_orig, ax_result) = plt.subplots(1, 2, figsize=(15,6))
    # fig.suptitle("Original (L) vs Result (R)")
    # ax_orig.imshow(x, cmap='hsv')
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()
    # ax_result.imshow(y, cmap='hsv')
    # ax_result.set_title('Result')
    # ax_result.set_axis_off()
    # fig.show()
    
    return y

def solve_63613498(x):
    #Assumptions: 
    #1. Background is always zero.
    #   Could use a histogram to identify the most common cell number (which would give 0 as a background)
    #   A problem with using a histogram is that the background may not necessarily be the most common cell value. 
    #   Also, we have not been told that background can change from test to test. Hence why I'm making the "background = 0" assumption.  
    #
    #2. The square at the top of the tile is fixed in position and size (ie 3x3) for all problems.
    #3. There is only one match. If there are multiple equivalent results, we just ignore all but the first "best" shape encountered.
    #4. Template shapes can be assumed to occupy the full area inside the bounds on each axis. ie templates are always 3x3.

    
    #Making global for debugging
    #global y
    #Define the template we're looking for.
    global template_zone_size 
    template_zone_size = (4,4)
    
    #As colour is not the same between template and the shape we're looking for, we will binarise the image so we can use convolution later to identify the best match to our template. 
    binary = binarised(x)
    
    #Won't bother cropping out background pixels from the template anymore, as it just caused some confusion.
    #As a result, this function just extract the template ratyher than also reduce it to its minimum size. 
    #eg  an L shape would only occupy 2 colummns, so we could crop out one column. Anyway, I implemented it and then it wasn't really needed, so I commented it out. 
    template = identifyAndCrop(binary)
    
    #I think a small bug came in when I switched from using convolution to correlation for the template matching.
    #Not sure what the problem was here, but had to make a copy for the pipeline to work.  Didn't have time to investigate it fully.
    template_copy = np.copy(template)
    
    #"Zero-out" the template and boundary in the original image so our correlation doesnt run on it and generate a match from itself.
    binary[0:template_zone_size[0],0:template_zone_size[1]] = np.zeros(template_zone_size)
    #Convolve the template over the image to create a heatmap of the best matches.
    #y = signal.convolve2d(binary, template_flipped, boundary='symm', fillvalue=0, mode='same')
    y = signal.correlate2d(binary, template_copy, boundary='symm', fillvalue=0, mode='same')
    
    
    #Find the image location with the max value from the convolution.
    L = np.unravel_index(np.argmax(y),np.shape(y))
    
    
    #Identify the colour / number value to assign to the pixels that make up the best match.
    #These always match the boundary of the 3x3 grid at the top left of the image.
    colour = x[template_zone_size[0]-1,template_zone_size[1]-1]
    #print("Colour is: ",colour)
    #Replace the match in the original image with our template (including an updated colour to match the boundary).
    #start_row = L[0] - np.shape(template)[0]
    #start_col = L[1] - np.shape(template)[1]
    #x[start_row:start_row+template_zone_size[0],start_col:start_col+template_zone_size[1]] = 10
    y = np.copy(x)
    y[L[0]-1:L[0]+2,L[1]-1:L[1]+2] = np.multiply(binary[L[0]-1:L[0]+2,L[1]-1:L[1]+2],colour)
    
    #Plot result.
    #print("binary: ", binary)
    #print("Template: ", template)
    #print("Template flipped: ", template_flipped)
    #print("y: ", y)
    #print("Location: ", L)
    #print("Template shape: ", np.shape(template))
    
    ##Debug prints
    # fig, (ax_orig, ax_result) = plt.subplots(2, 1, figsize=(6, 15))
    # ax_orig.imshow(x, cmap='hsv')
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()
    # ax_result.imshow(y, cmap='gray')
    # ax_result.set_title('Result')
    # ax_result.set_axis_off()
    # fig.show()
    return y

def solve_b8825c91(x):
    #Make a copy of x, so we can overwrite the corrupted values as we identify the correct values for a given pixel.    
    y = np.copy(x)
    
    #Determine the shape of the iomage so we can get an idea (even by eye) who to handle coordinate offsets from the center of the image. 
    s = np.shape(x)
    
    #Iterate through each pixel in the image, and identify a good value for that pixel.
    #Due to some pixels being corrupted, some pixels will need to be edited.
    for r in range(s[0]):
        for c in range(s[1]):
            #Identify the best replacement pixel value for the current one, and replace it.
            y[r,c] = getBestReplacement(r,c,y)

    # #Some debug images.
    # fig, (ax_orig, ax_result) = plt.subplots(1, 2, figsize=(15,6))
    # fig.suptitle("Original (L) vs Result (R)")
    # ax_orig.imshow(x, cmap='hsv')
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()
    # ax_result.imshow(y, cmap='hsv')
    # ax_result.set_title('Result')
    # ax_result.set_axis_off()
    # fig.show()
    
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

