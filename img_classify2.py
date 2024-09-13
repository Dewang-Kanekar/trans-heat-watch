import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Paths for training, validation, and testing datasets
data_train_path = 'transform_images/train'
data_test_path = 'transform_images/test'
data_val_path = 'transform_images/validation'

# Image dimensions
img_width = 240
img_height = 240
batch_size = 32

# Data augmentation to artificially increase dataset size
data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# Load datasets
data_train = image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=batch_size,
    validation_split=False
)  
data_cat = data_train.class_names

data_val = image_dataset_from_directory(
    data_val_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    validation_split=False
)

data_test = image_dataset_from_directory(
    data_test_path,
    image_size=(img_height, img_width),
    shuffle=False,
    batch_size=batch_size,
    validation_split=False
)

# Preprocessing with augmentation and normalization
AUTOTUNE = tf.data.AUTOTUNE
data_train = data_train.map(lambda x, y: (data_augmentation(x), y))
data_train = data_train.prefetch(buffer_size=AUTOTUNE)
data_val = data_val.prefetch(buffer_size=AUTOTUNE)

# Load a pretrained model (e.g., MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(data_cat), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for early stopping and learning rate adjustment
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train the model
epochs_size = 50  # Increase number of epochs for better performance
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size, callbacks=[early_stopping, lr_scheduler])

# Plot accuracy and loss curves
epochs_range = range(len(history.history['accuracy']))
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

image = 'mix.png'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

print('Transformer in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))