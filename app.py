import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load model
model = load_model('hair_classification_model.h5')  # Ganti dengan path model Anda

# Class names (sesuai dengan model Anda)
class_names = ['Straight', 'Wavy', 'Curly']  # Sesuaikan nama kelas

# Function untuk memproses gambar
def process_image(image):
    image = image.resize((128, 128))  # Sesuaikan ukuran dengan model Anda
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension
    return image_array

# Function untuk prediksi
def predict(image):
    processed_image = process_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# UI Streamlit
st.title("Hair Type Classification")
st.write("Upload an image to classify the hair type!")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Buka gambar menggunakan PIL
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediksi jika tombol ditekan
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            predicted_class, confidence = predict(image)
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
