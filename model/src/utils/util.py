from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
   

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    
    if debug:
        cv2.imwrite("output/thresh.jpg", thresh)
        print("Saving Thresh image to disk")

    # find contours in the thresholded image and sort them by size in
    # descending order (for finding the corners)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
            "Try debugging your thresholding and contour steps."))
        
    if debug:
        # draw the contour of the puzzle on the image
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)

        cv2.imwrite("output/puzzle.jpg", output)
        print("Saving Puzzle with corners and edges image to disk")
        
    # top-down bird's eye view for the extracted image 
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2)) # Gray-scaled 
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2)) # Color image (might be useful later )

    if debug:
        cv2.imwrite("output/puzzle_transform.jpg", puzzle)
        print("Saving Transformed image to disk")

    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)

def extract_digit(cell, num:str ,debug=True):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imwrite(f"output/Cell-Thresh{num}.jpg", thresh)
    thresh = clear_border(thresh)

    if debug:
        cv2.imwrite(f"output/Cell-Thresh-clear-border{num}.jpg", thresh)


    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        print("Return None, could not find any contours")
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # Checking that the image contains some "useful" info
    if percentFilled < 0.005:
        print(f"Failed through the 0.5% check")
        return None
    
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imwrite(f"output/Digit{num}.jpg", digit)
        print("Saving Digit: {num}'s image to disk")

    return digit

def identify_image(model, image, debug=True):
    (puzzleImage, warped) = find_puzzle(image, True)

    board = np.zeros((9, 9), dtype="int")
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    cellLocs = []
    
    for y in range(0, 9):
        row = []
        
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            
            # crop the cell from the warped transform image and then extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            
            num = str(y)+str(x)
            print("Checking x and y: ", num)
            digit = extract_digit(cell, num, debug=True)
            
            if digit is not None:
                if debug:
                    print("Going for the predict on y: {y} & x: {x}")
                pred = model.predict(digit,False)
                print(f"Predicted for y: {y} & x: {x} = {pred}")
                board[y, x] = pred

        cellLocs.append(row)
        
    return board