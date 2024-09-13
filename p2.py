import tkinter as tk  
from tkinter import filedialog  
from PIL import Image, ImageTk  
  
class GUI:  
   def _init_(self, master):  
      self.master = master  
      self.master.title("My GUI Application")  
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
  
   def search_images(self):  
      # TO DO: implement image search logic here  
      # For now, just display a sample image  
      image_path = "C:\\Users\\dkkan\\Downloads\\project1\\project1\\Image_classify.keras"  
      image = Image.open(image_path)  
      photo = ImageTk.PhotoImage(image)  
      self.hot_image_label.config(image=photo)  
      self.hot_image_label.image = photo  
if __name__ == "__main__":  
  root = tk.Tk()  
  gui = GUI(root)
  root.mainloop()