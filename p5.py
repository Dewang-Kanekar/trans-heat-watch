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
        self.search_entry.pack(side=tk.TOP, pady=10)

        # Create search button
        self.search_button = tk.Button(self.frame, text="Upload Image", command=self.search_images)
        self.search_button.pack(side=tk.TOP, pady=10)

        # Frame for image categories
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(pady=20)

        # Create labels for each category with sample images
        self.hot_image_label = self.create_image_label(self.image_frame, "Hot Image", "C:\\Users\\dkkan\\Downloads\\project1\\project1\\transform_images\\validation\\hot_thermal images\\bosonpluscapture2.jpeg", 0)
        self.cool_image_label = self.create_image_label(self.image_frame, "Cool Image", "C:\\Users\\dkkan\\Downloads\\project1\\project1\\transform_images\\validation\\cool_thermal_images\\The-thermogram-shows-the-wiring-connections-in-three-current-transformers-Un-110-kV_Q320.jpg", 1)
        self.short_circuit_label = self.create_image_label(self.image_frame, "Short Circuit", "C:\\Users\\dkkan\\Downloads\\project1\\project1\\transform_images\\validation\\short_ciruit_img\\hqdefault.jpg", 2)
        self.fire_label = self.create_image_label(self.image_frame, "Fire", "C:\\Users\\dkkan\\Downloads\\project1\\project1\\transform_images\\validation\\fire_img\\main-qimg-b5d84b81e2d2a474ade26902b40214d6-pjlq.jpg", 3)

        # Load the model
        self.model = tf.keras.models.load_model('C:\\Users\\dkkan\\Downloads\\project1\\project1\\Image_classify.keras')
        self.data_cat = ['cool_thermal_images', 'fire_img', 'hot_thermal images', 'short_ciruit_img']

    def create_image_label(self, parent, text, image_path, column):
        frame = tk.Frame(parent)
        frame.grid(row=0, column=column, padx=20)
        label = tk.Label(frame, text=text)
        label.pack()
        image = Image.open(image_path)
        image.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(frame, image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        image_label.pack()
        accuracy_label = tk.Label(frame, text="Accuracy: N/A")
        accuracy_label.pack()
        return image_label, accuracy_label

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

        # Update the corresponding image display and accuracy
        image = Image.open(image_path)
        image.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(image)
        if class_name == 'hot_thermal images':
            self.hot_image_label[0].config(image=photo)
            self.hot_image_label[0].image = photo
            self.hot_image_label[1].config(text=f"Accuracy: {accuracy:.2f}%")
        elif class_name == 'cool_thermal_images':
            self.cool_image_label[0].config(image=photo)
            self.cool_image_label[0].image = photo
            self.cool_image_label[1].config(text=f"Accuracy: {accuracy:.2f}%")
        elif class_name == 'short_ciruit_img':
            self.short_circuit_label[0].config(image=photo)
            self.short_circuit_label[0].image = photo
            self.short_circuit_label[1].config(text=f"Accuracy: {accuracy:.2f}%")
        elif class_name == 'fire_img':
            self.fire_label[0].config(image=photo)
            self.fire_label[0].image = photo
            self.fire_label[1].config(text=f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
