import numpy as np
from PIL import Image

def prepare_image(file):
    """ Convert the uploaded image file to an appropriate format for the model """
    image = Image.open(file)
    image = image.resize((256, 256))  # Adjust size to your model's requirements
    image = np.array(image)
    image = image / 255.0  # Normalize if your model expects this
    image = np.expand_dims(image, axis=0)
    return image
