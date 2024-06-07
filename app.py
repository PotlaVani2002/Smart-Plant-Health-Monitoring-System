from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Load the model
model_path = "E:/Projects/Smart-Plant-Health-Monitoring-System/model.h5"
loaded_model_hdf5 = load_model(model_path)
image_size = 256

class_names = {
    0: "Pepper__bell___Bacterial_spot", 1: "Pepper__bell___healthy", 2: "Potato___Early_blight",
    3: "Potato___Late_blight", 4: "Potato___healthy", 5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blightg", 7: "Tomato_Late_blight", 8: "Tomato_Leaf_Mold", 9: "Tomato_Septoria_leaf_spot",
    10:"Tomato_Spider_mites_Two_spotted_spider_mite",11: "Tomato__Target_Spot",12:"Tomato__Tomato_YellowLeaf__Curl_Virus", 
    13:"Tomato__Tomato_mosaic_virus",14:"Tomato_healthy",15:"cotton"
}

def process_image(img):
    img = tf.image.resize(img, (image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist('file')
        predictions = []

        for file in files:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = tf.convert_to_tensor(img)
            img = process_image(img)

            # Make prediction
            predictions_hdf5 = loaded_model_hdf5.predict(img)
            predicted_class_hdf5 = class_names[np.argmax(predictions_hdf5[0])]

            predictions.append(predicted_class_hdf5)

        return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)