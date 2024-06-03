from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

def load_model():
    # Load your TensorFlow model here
    return tf.keras.models.load_model('modelheart.keras',compile=True)

def prepare_image(file):
    # Convert the image file to an appropriate numpy array
    img = Image.open(file.stream).convert('L')  # Assuming grayscale images
    img = img.resize((256, 256))  # Resize to the expected input size of the model
    img = np.array(img)/255.0
    return img[np.newaxis, :, :, np.newaxis]  # Add batch dimension

def encode_image(image):
    # Ensure the image array is in the correct format
    if image.ndim == 3 and image.shape[2] == 1:  # Check if the image has a single channel
        image = image.squeeze(axis=2)  # Remove the channel dimension if it's single-channel
    elif image.ndim == 4 and image.shape[3] == 1:  # Check if it's a batch of single-channel images
        image = image.squeeze(axis=3)  # Remove the channel dimension

    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Assuming the image data ranges between 0 and 1

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded_image}'

app = Flask(__name__)
model = load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No file uploaded.'}), 400
    #file = img = Image.open("F:\'Heart segmentation'\test\test2.PNG")
    image = prepare_image(file)
    original, mask = predict(image, model)
    # Encode the images to send as JSON
    encoded_original = encode_image(original)
    encoded_mask = encode_image(mask)
    return jsonify({'original': encoded_original, 'mask': encoded_mask})

def predict(image, model):
    predictions = model.predict(image)
    
    print(predictions.max())
    # Threshold to create a binary mask
    mask = (predictions[0, :, :, 0]).astype(np.float32)/predictions.max()  # Assuming the model outputs single-channel prediction
    return image[0, :, :, 0], mask  # Return the first image and its mask from batch

if __name__ == '__main__':
    app.run(debug=True)
