import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Transformer Image Classification")
        self.frame = tk.Frame(self.master)
        self.frame.pack()

        # Create search entry field
        self.search_entry = tk.Entry(self.frame, width=30)
        self.search_entry.pack(side=tk.LEFT)

        # Create search button
        self.search_button = tk.Button(self.frame, text="Search", command=self.search_images)
        self.search_button.pack(side=tk.LEFT)

        # Create image display labels
        self.hot_image_label = tk.Label(self.frame, text="Hot Image")
        self.hot_image_label.pack()
        self.cool_image_label = tk.Label(self.frame, text="Cool Image")
        self.cool_image_label.pack()
        self.short_circuit_label = tk.Label(self.frame, text="Short Circuit")
        self.short_circuit_label.pack()
        self.fire_label = tk.Label(self.frame, text="Fire")
        self.fire_label.pack()

        # Load the model
        self.model = tf.keras.models.load_model('C:\\Users\\dkkan\\Downloads\\project1\\project1\\Image_classify.keras')
        self.data_cat = ['cool_thermal_images', 'fire_img', 'hot_thermal images', 'short_ciruit_img']

    def search_images(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.classify_image(file_path)

    def classify_image(self, image_path):
        img_height = 180
        img_width = 180
        image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        predict = self.model.predict(img_bat)
        score = tf.nn.softmax(predict[0])
        class_name = self.data_cat[np.argmax(score)]
        accuracy = np.max(score) * 100

        # Update image display
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        if class_name == 'hot_thermal images':
            self.hot_image_label.config(image=photo, text=f"Hot Image: {accuracy:.2f}%")
            self.hot_image_label.image = photo
        elif class_name == 'cool_thermal_images':
            self.cool_image_label.config(image=photo, text=f"Cool Image: {accuracy:.2f}%")
            self.cool_image_label.image = photo
        elif class_name == 'short_ciruit_img':
            self.short_circuit_label.config(image=photo, text=f"Short Circuit: {accuracy:.2f}%")
            self.short_circuit_label.image = photo
        elif class_name == 'fire_img':
            self.fire_label.config(image=photo, text=f"Fire: {accuracy:.2f}%")
            self.fire_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
