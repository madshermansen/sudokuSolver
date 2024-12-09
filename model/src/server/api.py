from flask import Flask, request, jsonify
from utils.util import identify_image
import numpy as np
import cv2
from sudoku import Sudoku

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
        result = identify_image(model=model,image=image,debug=False)

        print("Puzzle found!")
        print("result: ",result)

        grid = result.tolist()
        solved_by_model = create_solved_by_model_grid(grid)
        puzzle = Sudoku(3, 3, board=grid)
        solved = True
        solved_board = puzzle
        try:
            puzzle.show_full()
            print("Solving...")
            solved_board = puzzle.solve(rasing=True)
            print("Solved!")
        except:
            print("Could not solve the puzzle.")
            solved = False


        # Placeholder response
        response = {
            "message": "Image received successfully",
            'data': {
                'sudokuAnswer': solved_board.board,
                'solvedByModel': solved_by_model,
                'solved': solved

            },
            "filename": image_file.filename,
        }
        

        print('Sending response =======================>\n', response)
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

def create_solved_by_model_grid(grid):
    solved_by_model = []
    for row in grid:
        solved_by_model.append([1 if x == 0 else 0 for x in row])
    return solved_by_model