from flask import Flask, request, jsonify
from utils.util import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
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
    

        # Placeholder response
        response = {
            "message": "Image received successfully",
            "filename": image_file.filename,
        }
        
        # Read the image directly from memory
        
        image = cv2.imread("uploaded_image.jpg")

        if image is None:
            return jsonify({"error": "Could not process the image."}), 400

        print("Finding puzzle...")
        
        (puzzleImage, warped) = find_puzzle(image, True)
        # initialize our 9x9 Sudoku board
        board = np.zeros((9, 9), dtype="int")
        # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
        # infer the location of each cell by dividing the warped image
        # into a 9x9 grid
        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9
        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cellLocs = []
        
        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                # add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))
                
                # crop the cell from the warped transform image and then
                # extract the digit from the cell
                cell = warped[startY:endY, startX:endX]
                num = str(y)+str(x)
                digit = extract_digit(cell, num, debug=True)
                print("Digit is: ", num)
                # # verify that the digit is not empty
                # if digit is not None:
                #     # resize the cell to 28x28 pixels and then prepare the
                #     # cell for classification
                #     roi = cv2.resize(digit, (28, 28))
                #     roi = roi.astype("float") / 255.0
                #     roi = img_to_array(roi)
                #     roi = np.expand_dims(roi, axis=0)
                #     # classify the digit and update the Sudoku board with the
                #     # prediction
                #     pred = model.predict(roi).argmax(axis=1)[0]
                #     board[y, x] = pred
            # add the row to our cell locations
            cellLocs.append(row)
            
        print(board)
        print("Puzzle found!")

        # Use the model to make predictions
        result = model.predict("uploaded_image.jpg")
        print(result)

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
