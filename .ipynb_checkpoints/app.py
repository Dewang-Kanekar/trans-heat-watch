import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.models import  load_model # type: ignore
import streamlit as st # type: ignore
import numpy as np  # type: ignore

st.header('Image Classification Model')
model = load_model('C:\\Users\\purva\\Downloads\\project1\\Image_classify.keras')
data_cat = ['cool_thermal_images', 
            'fire_img', 
            'hot_thermal images', 
            'short_ciruit_img']
img_height = 180
img_width = 180
image =st.text_input('Enter Image name','a3.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Transformer in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))
import tkinter as tk
from tkinter import ttk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("4 Columns and 1 Dial Example")

        # Create a Frame for the layout
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create 4 Columns
        for i in range(4):
            column_label = ttk.Label(self.frame, text=f"Column {i+1}")
            column_label.grid(row=0, column=i, padx=5, pady=5, sticky=tk.W)

        # Add some widgets to each column for demonstration
        for i in range(4):
            entry = ttk.Entry(self.frame)
            entry.grid(row=1, column=i, padx=5, pady=5, sticky=(tk.W, tk.E))

        # Create a dial (using a Scale widget for simplicity)
        self.dial = tk.Scale(self.frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Dial")
        self.dial.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky=(tk.W, tk.E))

# Create the main application window
root = tk.Tk()
app = App(root)

# Run the application
root.mainloop()
