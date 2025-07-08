import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import datetime

# --------------------- Page Config ---------------------
st.set_page_config(page_title="Garbage Classifier", layout="centered")

# --------------------- SEO Tags (optional) ---------------------
st.markdown(
    """
    <meta name="description" content="Garbage classification app using AI. Upload or capture trash images to detect type (plastic, glass, etc).">
    <meta name="keywords" content="garbage classification, trash detection, AI waste sorting, deep learning streamlit app">
    <meta name="author" content="Your Name">
    <meta property="og:title" content="Garbage Classifier App">
    <meta property="og:description" content="Upload, capture, or paste a link to classify garbage images using deep learning.">
    """,
    unsafe_allow_html=True
)

# --------------------- Class Labels ---------------------
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --------------------- Load the Model ---------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# --------------------- Image Preprocessing ---------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------- App Header ---------------------
st.title(" Garbage Classification App")
st.markdown("Upload an image, take a photo, or paste a URL to classify the type of garbage using deep learning.")

# --------------------- Sidebar: Select Input Method ---------------------
with st.sidebar:
    st.header(" Choose Input Method")
    input_method = st.radio("Select image input source:", [
        "Upload File",
        "Camera",
        "Image URL",
        "Use Example Image"
    ])

st.info("Tip: You can upload, capture, paste a URL, or choose a built-in example to classify.")

# --------------------- Load Image ---------------------
uploaded_image = None

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)

elif input_method == "Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        uploaded_image = Image.open(camera_image)

elif input_method == "Image URL":
    image_url = st.text_input("Paste image URL here:")
    if image_url:
        try:
            response = requests.get(image_url)
            uploaded_image = Image.open(BytesIO(response.content))
        except:
            st.error(" Could not load image from URL.")

elif input_method == "Use Example Image":
    example_choice = st.selectbox("Choose example image:", ["plastic", "glass", "metal"])
    if example_choice:
        try:
            uploaded_image = Image.open(f"examples/{example_choice}.jpg")
        except:
            st.error(" Example image not found. Make sure it exists in 'examples/' folder.")

# --------------------- Perform Prediction ---------------------
if uploaded_image:
    st.image(uploaded_image, caption="Input Image", use_column_width=True)

    with st.spinner(" Classifying... Please wait..."):
        img_array = preprocess_image(uploaded_image)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

    st.success(f" Predicted Class: **{predicted_class}**")
    st.info(f" Confidence Score: **{confidence:.2f}**")

    # Progress Bar
    st.progress(int(confidence * 100))

    # Bar Chart of All Predictions
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0], color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Download Result
    result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}\nTime: {datetime.datetime.now()}"
    st.download_button(" Download Prediction", result_text, file_name="prediction_result.txt")

    # Save Prediction History in Session State
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append((predicted_class, confidence))

    # Show History (last 5)
    if st.checkbox(" Show Prediction History"):
        for i, (pc, conf) in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.write(f"{i}. {pc} – {conf:.2f}")

    # Reset Button
    if st.button(" Reset"):
        st.rerun()
