import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 48x48
    image = image.resize((48, 48))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape the image to match the model's input shape
    image = np.reshape(image, (1, 48, 48, 1))
    return image

# Streamlit app
st.title("Emotion Detection App")
st.write("Upload an image, and the model will predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    
    # Display the prediction
    st.write(f"Predicted Emotion: *{predicted_emotion}*")
    st.write("Prediction Probabilities:")
    for i, emotion in enumerate(emotion_labels):
        st.write(f"{emotion}: {prediction[0][i]:.4f}")