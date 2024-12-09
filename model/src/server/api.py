from flask import Flask, request, jsonify
from utils.util import identify_image
import numpy as np
import cv2

app = Flask(__name__)

model = None

@app.route('/api/solve', methods=['POST'])
def solve_image():
    try:
        # Check if the 'image' field exists in the files
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save the uploaded image temporarily (or process it directly)
        image_file.save("uploaded_image.jpg")
        print(f"Image saved as: uploaded_image.jpg")
    
        # Read the image directly from memory
        
        image = cv2.imread("uploaded_image.jpg")

        if image is None:
            return jsonify({"error": "Could not process the image."}), 400

        print("Finding puzzle...")
        result = identify_image(model=model,image=image,debug=True)

        print("Puzzle found!")
        print("result: ",result)


        grid = result

        solved_by_model = create_solved_by_model_grid(grid)
        solveSudoku(grid)

        # Placeholder response
        response = {
            "message": "Image received successfully",
            'data': {
                'sudokuAnswer': grid,
                'solvedByModel': solved_by_model,
            },
            "filename": image_file.filename,
        }
        

        return jsonify(response), 200

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

def run_server(model_instance=None):
    global model
    model = model_instance

    if model is None:
        print("Model is not provided!")
        return

    app.run(debug=True, host='0.0.0.0', port=8080)


def findNextCellToFill(grid, i, j):
        for x in range(i,9):
                for y in range(j,9):
                        if grid[x][y] == 0:
                                return x,y
        for x in range(0,9):
                for y in range(0,9):
                        if grid[x][y] == 0:
                                return x,y
        return -1,-1

def isValid(grid, i, j, e):
        rowOk = all([e != grid[i][x] for x in range(9)])
        if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)])
                if columnOk:
                        # finding the top left x,y co-ordinates of the section containing the i,j cell
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
                        for x in range(secTopX, secTopX+3):
                                for y in range(secTopY, secTopY+3):
                                        if grid[x][y] == e:
                                                return False
                        return True
        return False

def solveSudoku(grid, i=0, j=0):
        i,j = findNextCellToFill(grid, i, j)
        if i == -1:
                return True
        for e in range(1,10):
                if isValid(grid,i,j,e):
                        grid[i][j] = e
                        if solveSudoku(grid, i, j):
                                return True
                        # Undo the current cell for backtracking
                        grid[i][j] = 0
        return False

def create_solved_by_model_grid(grid):
    solved_by_model = []
    for row in grid:
        solved_by_model.append([1 if x != 0 else 0 for x in row])
    return solved_by_model