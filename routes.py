import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# This looks one folder UP from 'Apps' for the model file
MODEL_PATH = os.path.join(BASE_DIR, '..', 'crowd_model.h5') 

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("AI Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: crowd_model.h5 not found in root folder.")

def generate_heatmap(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 0
    
    # Pre-processing for CNN-LSTM
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(processed_img, (238, 158)) / 255.0
    input_seq = np.array([[processed_img] * 10]) # Sequence of 10
    input_seq = np.expand_dims(input_seq, -1)

    if model:
        prediction = model.predict(input_seq)
        count = int(np.round(prediction[0][0]))
    else:
        count = "Model not trained"

    return count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    
    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    file.save(filepath)
    count = generate_heatmap(filepath)

    # We only return the count now, no image URLs sent to the template
    return render_template('index.html', crowd_count=count)

if __name__ == '__main__':
    app.run(debug=True)