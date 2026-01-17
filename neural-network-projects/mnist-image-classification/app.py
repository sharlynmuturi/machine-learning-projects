import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0–9) or upload an image to get a prediction.")

# Loading the model
@st.cache_resource
def load_model():
    model_path = os.path.join("neural-network-projects", "mnist-image-classification", "cnn_digits.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()


# Pre-Processing Image
def crop_and_center(img):
    gray = img.copy()
    # Invert colors so digit is white
    gray = np.invert(gray)

    # Threshold img - convert to binary; pixels > 20 - 255 (white), else 0 (black)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y:y+h, x:x+w]

    # Make square
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    return square


def preprocess_image(img):
    """
    Preprocess canvas/upload image for MNIST model
    """
    # Convert to grayscale
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop and center digit
    img = crop_and_center(img)

    # Resize to MNIST size
    img = cv2.resize(img, (28, 28))

    # Smooth jagged edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize
    img = img / 255.0

    # Reshape for model: (batch, height, width, channels)
    img = img.reshape(1, 28, 28, 1)

    return img


# Tabs for input methods
tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

# Drawing Tab
with tab1:
    st.subheader("Draw a digit")

    col1, col2 = st.columns([1, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=6,
            stroke_color="black",
            background_color="white",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    if canvas_result and canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        with col2:
            st.markdown("### Prediction")
            st.write(f"**Digit:** `{digit}`")
            st.write(f"**Confidence:** `{confidence:.2%}`")
            st.bar_chart(prediction[0])

# Upload tab
with tab2:
    st.subheader("Upload a handwritten digit image")

    uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=200)

        img = np.array(image)
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown("### Prediction")
        st.write(f"**Digit:** `{digit}`")
        st.write(f"**Confidence:** `{confidence:.2%}`")

        st.bar_chart(prediction[0])
