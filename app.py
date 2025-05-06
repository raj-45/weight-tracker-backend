from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
import cv2
import numpy as np
from datetime import datetime
import os
import requests

app = Flask(__name__)
CORS(app)

# Load environment variables from Render
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY")

# OCR function
def extract_weight_from_image(image_bytes):
    # Convert image bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale, preprocess if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Try to extract a float from the detected text
    for part in text.split():
        try:
            weight = float(part)
            return weight
        except ValueError:
            continue
    return None

# Save to Supabase
def save_to_supabase(weight):
    if not (SUPABASE_URL and SUPABASE_API_KEY):
        return False

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/weights",
        headers={
            "apikey": SUPABASE_API_KEY,
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        },
        json={
            "weight": weight,
            "date": datetime.utcnow().date().isoformat()
        }
    )
    return response.status_code == 201

# Routes
@app.route('/')
def home():
    return 'Weight Tracker API is live!'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    weight = extract_weight_from_image(image_bytes)

    if weight is None:
        return jsonify({'error': 'Could not read weight'}), 422

    saved = save_to_supabase(weight)
    return jsonify({'weight': weight, 'saved': saved})

if __name__ == '__main__':
    app.run(debug=True)
