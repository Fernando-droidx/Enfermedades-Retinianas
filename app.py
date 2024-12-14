from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


model = load_model('modelo/modelo_retiniano_gpu.h5')
class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Verificar si se envió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            return jsonify({'prediction': predicted_class})
        return jsonify({'error': 'Unknown error occurred'}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
