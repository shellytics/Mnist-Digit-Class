import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load model
model = load_model('mnist_cnn_model.keras')

st.title("MNIST Digit Classifier")
st.header("Upload an image of a handwritten digit")



def predict_digit(image_file):
    """Preprocess image and return prediction"""
    image = Image.open(image_file).convert('L')  # Grayscale
    image = image.resize((28, 28))               # Resize to MNIST format
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)   # Add batch and channel dims

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    return image, predicted_digit

# Upload image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image, digit = predict_digit(uploaded_file)

    st.image(image, caption="Uploaded Image", width=150)
    st.write(f"### Predicted Digit: {digit}")
