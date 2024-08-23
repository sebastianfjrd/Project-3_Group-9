from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your model (ensure the path is correct and accessible)
model = tf.keras.models.load_model('saved_model/melanoma_cnn_model.keras')

def format_image_for_prediction(image):
    image = image.resize((300, 300))
    image_normalized = np.array(image).astype(np.float32) / 255
    image_normalized = np.expand_dims(image_normalized, axis=0)
    return image_normalized

def get_prediction(image_normalized):
    prediction_probs = model.predict(image_normalized)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    confidence = np.max(prediction_probs)
    return {'prediction': prediction, 'confidence': float(confidence)}

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    image = Image.open(file.stream)
    image_normalized = format_image_for_prediction(image)
    result = get_prediction(image_normalized)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
