from flask import Flask, request, jsonify

app = Flask(__name__)

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
        return jsonify(response), 200

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
