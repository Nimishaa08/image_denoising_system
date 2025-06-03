import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model.h5')

st.title("🧹 Image Denoising using Autoencoder")
st.write("Upload a **noisy 28x28 grayscale image**, and the model will denoise it.")

# Upload the image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))                  # Resize for model input
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image) / 255.0             # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)      # Reshape for model

    # Predict / Denoise
    denoised = model.predict(img_array)
    denoised_image = denoised.reshape(28, 28)        # Remove extra dimensions

    # Show result
    st.subheader("🔧 Denoised Image")
    st.image(denoised_image, width=150, clamp=True)
