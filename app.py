import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

app = Flask(__name__)

# Define upload folder path
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
'                                                                                           '
# Load your trained model once when the app starts
model = load_model('model/plant_disease_model.h5')

# Paste your exact class names here (order must match training)
class_names = [


    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
    # add the rest of your classes exactly
]

def preprocess_image(image_path):
    """Load and preprocess image to feed into model"""
    img = load_img(image_path, target_size=(224, 224))  # load and resize
    img_array = img_to_array(img) / 255.0  # scale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess uploaded image
        image = preprocess_image(filepath)

        # Predict disease
        preds = model.predict(image)  # prediction probabilities
        class_idx = np.argmax(preds)  # get index of max prob
        predicted_disease = class_names[class_idx]
        probability_score = round(np.max(preds) * 100, 2)

        return render_template('result.html', disease_name=predicted_disease, probability=probability_score)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
