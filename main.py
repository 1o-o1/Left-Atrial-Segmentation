from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def load_model():
    return tf.keras.models.load_model('modelheart2.keras', compile=True)


def load_classifer():
    return tf.keras.models.load_model('modelCNNPred.keras', compile=True)

def prepare_image(file):
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.asarray(img)
    grayscale_img = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    img_array = np.array(grayscale_img, dtype=np.float32) / 255.0 * 6.8
    img_array = img_array[:, :, np.newaxis]
    return img_array[np.newaxis, ...]

def encode_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    image_png = buf.getvalue()
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

app = Flask(__name__)
model = load_model()
classifer = load_classifer()
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    image = prepare_image(file)
    predictions = model.predict(image)
    predictions = predictions / predictions.max()  # Normalize the predictions
    
    # Plot the original and predicted mask using Matplotlib for visualization
    encoded_original = encode_image(image[0, :, :, 0])  # Encode the original image
    encoded_mask = encode_image(predictions[0, :, :, 0])  # Encode the predicted mask
    class_pred =classifer.predict(image)
    prediction_confidence = np.mean(class_pred)
    
    return jsonify({'original': encoded_original, 'mask': encoded_mask, 'prediction': str(prediction_confidence)})

if __name__ == '__main__':
    app.run(debug=True)
