import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os

def load_model():
    # Load and return TensorFlow model
    return tf.keras.models.load_model('modelheart2.keras', compile=True)

def load_image(file_path, target_size=(256, 256)):
    # Load an image file to an appropriate numpy array
    img = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize(target_size)  # Resize to match the input size of the model
    
    # Convert RGB to grayscale using luminance-preserving weights
    img_array = np.asarray(img)
    grayscale_img = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Normalize the image and convert to float32
    img_array = np.array(grayscale_img, dtype=np.float32) / 255.0 * 6.8
    img_array = img_array[:, :, np.newaxis]  # Add channel dimension at the end
    return img_array[np.newaxis, ...]  # Add batch dimension

def predict_and_plot(image_path, model):
    # Load image
    image = load_image(image_path)
    
    # Predict mask
    prediction = model.predict(image)
    
    # Plot the original image and the predicted mask
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()

def main():
    model = load_model()
    test_image_path = 'test/original_image_40.png'  # Update this to the path of the image you want to test
    predict_and_plot(test_image_path, model)

if __name__ == '__main__':
    main()
