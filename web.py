import cv2
import numpy as np
from flask import Flask, render_template, Response
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('Image_classify2.keras')

# Define image size and categories
img_width, img_height = 240, 240
categories = ['Hotspot', 'Short Circuit', 'Fire', 'Cool']  # Adjust as needed

# Flask app
app = Flask(__name__)

# Function to capture and process the video stream
def gen_frames():
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        success, frame = cap.read()  # Read frame from the camera
        if not success:
            break
        else:
            # Preprocess the frame for prediction
            img = cv2.resize(frame, (img_width, img_height))
            img_array = np.expand_dims(img, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize pixel values

            # Make predictions
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            label = categories[class_idx]  # Get the class label
            confidence = np.max(predictions) * 100  # Get the confidence in percentage
            confidence = min(confidence, 100.0)  # Cap confidence at 100%

            # Draw bounding box and label on the frame
            (h, w) = frame.shape[:2]
            startX, startY = 10, 10  # Starting point for bounding box (example values)
            endX, endY = w - 10, h - 10  # Ending point for bounding box (example values)
            
            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}%", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route for the home page
@app.route('/')
def index():
    return render_template('web2.html')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
