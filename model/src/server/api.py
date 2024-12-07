from flask import Flask, request, jsonify

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
            'data': {
                'sudokuAnswer': [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 [4, 5, 6, 7, 8, 9, 1, 2, 3],
                                 [7, 8, 9, 1, 2, 3, 4, 5, 6],
                                 [2, 3, 4, 5, 6, 7, 8, 9, 1],
                                 [5, 6, 7, 8, 9, 1, 2, 3, 4],
                                 [8, 9, 1, 2, 3, 4, 5, 6, 7],
                                 [3, 4, 5, 6, 7, 8, 9, 1, 2],
                                 [6, 7, 8, 9, 1, 2, 3, 4, 5],
                                 [9, 1, 2, 3, 4, 5, 6, 7, 8]],
                'solvedByModel': [[0,0,1,0,1,1,0,0,0],
                                 [0,0,1,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0,0],
                                 [0,0,0,0,1,0,0,0,0],
                                 [1,0,0,0,0,1,0,0,0],
                                 [1,0,0,0,0,0,0,0,1],
                                 [0,0,0,0,1,0,1,1,0],
                                 [0,0,0,1,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0,0]],
            },
            "filename": image_file.filename,
        }

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
