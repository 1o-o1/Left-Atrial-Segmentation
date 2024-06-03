import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def load_model():
    # Load your TensorFlow model here
    return tf.keras.models.load_model('modelheart.keras',compile=True)

def plot_predictions(original_image, prediction_mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction_mask, cmap='gray')
    axes[1].set_title('Prediction Mask')
    axes[1].axis('off')
    
    plt.show()

def prepare_image(file):
    # Convert the image file to an appropriate numpy array
    img = Image.open(file).convert('L')  # Assuming grayscale images
    img = img.resize((256, 256))  # Resize to the expected input size of the model
    img = np.array(img)/255.0
    return img[np.newaxis, :, :, np.newaxis]  # Add batch dimension

model = load_model()
img = prepare_image("test\test2.PNG")
predictions = model.predict(img)
plot_predictions(img[0], predictions[0])