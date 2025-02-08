# app.py
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from scipy.special import softmax
import pickle
import os
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Healthy', 'Rice blast', 'Sheath blight', 'Tungro', 'Unknown Image or Diseases']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODELS_DIR = 'models'
UPLOADS_DIR = 'uploads'
SOL_DIR = 'solutions'
BIC_DIR = 'bicol'
TAG_DIR = 'tagalog'
ENG_DIR = 'english'

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Load model and configuration files
try:
    with open(os.path.join(MODELS_DIR, 'class_means.pkl'), 'rb') as f:
        class_means = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'precision_matrix.pkl'), 'rb') as f:
        prec = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'rice_leaf_disease_model.keras'))
except Exception as e:
    logger.error(f"Error loading configuration files: {e}")
    class_means = None
    prec = None
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_mahalanobis(X, mean, prec):
    try:
        X_minus_mu = X - mean
        left = np.dot(X_minus_mu, prec)
        mahal = np.dot(left, X_minus_mu.T)
        return np.sqrt(mahal.diagonal())
    except Exception as e:
        logger.error(f"Error computing Mahalanobis distance: {e}")
        return None

def predict_disease(image_path, mahalanobis_threshold=50, entropy_threshold=0.5):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        logits = model.predict(img_array)[0]
        probabilities = softmax(logits)
        predicted_class = CLASS_NAMES[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        pred_entropy = entropy(probabilities)
        max_entropy = np.log(len(CLASS_NAMES))
        normalized_entropy = pred_entropy / max_entropy

        feature_model = tf.keras.Model(model.input, model.layers[-2].output)
        features = feature_model.predict(img_array)[0]

        mahalanobis_distances = [compute_mahalanobis(features.reshape(1, -1), class_means[i], prec)[0] for i in range(len(CLASS_NAMES))]
        min_mahalanobis = np.min(mahalanobis_distances)

        if min_mahalanobis > mahalanobis_threshold or normalized_entropy > entropy_threshold:
            return "Unknown Image or Diseases", confidence if confidence is not None else 0.0, normalized_entropy if normalized_entropy is not None else 0.0, min_mahalanobis if min_mahalanobis is not None else 0.0, probabilities
        else:
            return predicted_class, confidence if confidence is not None else 0.0, normalized_entropy if normalized_entropy is not None else 0.0, min_mahalanobis if min_mahalanobis is not None else 0.0, probabilities

    except Exception as e:
        logger.error(f"Error predicting disease: {e}")
        return "Unknown Image or Diseases", 0.0, 0.0, 0.0, [0.0] * len(CLASS_NAMES)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            filepath = os.path.join(UPLOADS_DIR, filename)
            file.save(filepath)

            prediction, confidence, entropy_val, mahalanobis, probabilities = predict_disease(filepath)

            # for bicol result
            solution_image_bic = None
            if prediction and prediction not in ['Healthy', 'Unknown Image or Diseases']:
                solution_path = os.path.join(SOL_DIR, BIC_DIR, f"{prediction}.png")
                if os.path.exists(solution_path):
                    with open(solution_path, 'rb') as img_file:
                        solution_image_bic = base64.b64encode(img_file.read()).decode()

            # for tagalog result
            solution_image_tag = None
            if prediction and prediction not in ['Healthy', 'Unknown Image or Diseases']:
                solution_path = os.path.join(SOL_DIR, TAG_DIR, f"{prediction}.png")
                if os.path.exists(solution_path):
                    with open(solution_path, 'rb') as img_file:
                        solution_image_tag = base64.b64encode(img_file.read()).decode()

            # for english result
            solution_image_eng = None
            if prediction and prediction not in ['Healthy', 'Unknown Image or Diseases']:
                solution_path = os.path.join(SOL_DIR, ENG_DIR, f"{prediction}.png")
                if os.path.exists(solution_path):
                    with open(solution_path, 'rb') as img_file:
                        solution_image_eng = base64.b64encode(img_file.read()).decode()

            os.remove(filepath)

            return jsonify({
                'prediction': prediction,
                'confidence': float(confidence),
                'entropy': float(entropy_val),
                'mahalanobis': float(mahalanobis),
                'probabilities': probabilities.tolist(),
                'solution_image_bic': solution_image_bic,
                'solution_image_tag': solution_image_tag,
                'solution_image_eng': solution_image_eng
            })

        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)