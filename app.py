import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import os
import numpy as np

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(working_dir, "D:\\desktop\\Image classification\\Image_classify_model.h5")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# List of categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

img_height = 180
img_width = 180

# Preprocess image
def preprocess_img(image):
    image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
    img_bat = tf.expand_dims(image_load, 0)
    return img_bat

# Streamlit app
st.title('Veg/Fruits Classification')
uploaded_image = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((300, 300))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                # Preprocess the uploaded image
                img_arr = preprocess_img(uploaded_image)
                # Make a prediction using the pre-trained model
                result = model.predict(img_arr)
                predicted_class = int(tf.argmax(result, axis=-1))
                prediction = data_cat[predicted_class]
                # Get the predicted probability for the predicted class
                predicted_probability = result[0][predicted_class]
                st.success(f'Prediction: {prediction}')
                st.write(f'Predicted Probability: {predicted_probability:.2f}')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
