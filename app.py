import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- KONFIGURASI ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model.h5'
IMG_SIZE = (128, 128)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model di awal agar respons lebih cepat
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
else:
    print("Model tidak ditemukan! Pastikan sudah menjalankan train_model.py")
    model = None

def preprocess_image(image_path):
    """
    Preprocessing citra sebelum diprediksi:
    1. Grayscale
    2. Resize ke 128x128
    3. Normalisasi
    4. Reshape agar sesuai input model (batch_size, height, width, channels)
    """
    img = load_img(image_path, color_mode='grayscale', target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Proses prediksi
        if model is not None:
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)[0][0]
            
            # 1 = Asli, 0 = Palsu
            result = "Asli (Genuine)" if prediction > 0.5 else "Palsu (Forged)"
            confidence = float(prediction if prediction > 0.5 else 1 - prediction) * 100
            
            return jsonify({
                'result': result,
                'confidence': f"{confidence:.2f}%",
                'image_url': filepath
            })
        else:
            return jsonify({'error': 'Model belum tersedia'})

if __name__ == '__main__':
    # Railway menyediakan port melalui environment variable PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
