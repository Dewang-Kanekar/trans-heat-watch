import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Image Classification Model')
model = load_model('Image_classify2.keras')
data_cat = ['cool_thermal_images', 
            'fire_img', 
            'hot_thermal images', 
            'short_ciruit_img']
img_height = 240
img_width = 240
image =st.text_input('Enter Image name','a1.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Transformer in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))