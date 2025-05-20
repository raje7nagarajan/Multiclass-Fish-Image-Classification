import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the best model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('D:/DA/Fish_IC/mobilenet_trained_model.h5')

model = load_model()

# Class names
class_names = ['fish sea_food trout', 'fish sea_food striped_red_mullet', 'fish sea_food shrimp', 'fish sea_food sea_bass', 'fish sea_food red_sea_bream', 'fish sea_food red_mullet', 'fish sea_food hourse_mackerel', 'fish sea_food gilt_head_bream', 'fish sea_food black_sea_sprat', 'animal fish', 'animal fish bass']

st.markdown(
    "<h1 style='text-align: center; color: orange;'>FISH CLASSIFIER</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='color: #8B4513;'>Upload a fish image to predict its species</h3>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    image_resized = image.resize((224, 224))  # Adjust based on model input
    image_array = np.array(image_resized) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_batch)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index]

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {class_names[pred_index]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Display confidence scores
    st.subheader("Confidence Scores")
    for i, score in enumerate(prediction):
        st.write(f"{class_names[i]}: {score:.2f}")
