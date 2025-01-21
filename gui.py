import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('./our-model1.h5')  # Replace with your model's path

# Define the class names
class_names = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Preprocessing function for the uploaded image
def preprocess_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))  # Adjust size to your model's input
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize if your model requires it
    return image_array

# Prediction function
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the uploaded image
        img = Image.open(file_path)
        img.thumbnail((250, 250))  # Adjust size for display
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        # Preprocess and predict
        image_array = preprocess_image(file_path)
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100

        # Display prediction
        result_label.config(text=f"Predicted Class: {predicted_class_name}\nConfidence: {confidence:.2f}%")

# GUI setup
root = tk.Tk()
root.title("Image Classifier")
root.geometry("400x400")

# Widgets
panel = Label(root)
panel.pack(pady=10)

upload_btn = Button(root, text="Upload Image", command=predict_image)
upload_btn.pack(pady=10)

result_label = Label(root, text="Prediction Result", font=("Arial", 12))
result_label.pack(pady=10)

# Start the GUI
root.mainloop()
