TF_ENABLE_ONEDNN_OPTS=0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    return tf.keras.models.load_model('modelheart.keras')

def plot_predictions(original_image, prediction_mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction_mask, cmap='gray')
    axes[1].set_title('Prediction Mask')
    axes[1].axis('off')
    
    plt.show()

def predict(image, model):
    predictions = model.predict(image)
    # Assuming your model outputs a mask that is the same size as the input
    mask = (predictions > 0.5).astype("uint8")  # Example thresholding

    # Plotting the predictions for debugging
    plot_predictions(image[0], predictions[0])  # Assuming image is batched and mask is the first prediction

    return image, mask




