import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('Image_classify.keras')
data_cat = ['cool_thermal_images', 'fire_img', 'hot_thermal images', 'short_ciruit_img']

def classify_image(image_path):
    img_height = 180
    img_width = 180
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    class_name = data_cat[np.argmax(score)]
    accuracy = np.max(score) * 100

    return class_name, accuracy

# Set up the layout
st.title("Transformer Image Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","bmp"])

# Sample images
sample_images = {
    "Hot Image":'hot.jpg',
    "Cool Image": "p2022.bmp",
    "Short Circuit": "cat.jpg",
    "Fire": "cat2.jpg"
}

# Display sample images
cols = st.columns(4)
for i, (label, path) in enumerate(sample_images.items()):
    with cols[i]:
        st.header(label)
        image = Image.open(path)
        st.image(image, use_column_width=True)
        st.write("Accuracy: N/A")

# If an image is uploaded
if uploaded_file:
    image_path = uploaded_file.name
    image = Image.open(uploaded_file)
    class_name, accuracy = classify_image(uploaded_file)

    # Update the corresponding category box with the new image and accuracy
    cols = st.columns(4)
    for i, (label, path) in enumerate(sample_images.items()):
        if class_name in label.lower():
            with cols[i]:
                st.header(label)
                st.image(image, use_column_width=True)
                st.write(f"Accuracy: {accuracy:.2f}%")
        else:
            with cols[i]:
                st.header(label)
                sample_image = Image.open(path)
                st.image(sample_image, use_column_width=True)
                st.write("Accuracy: N/A")
