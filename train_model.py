import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- KONFIGURASI ---
IMG_SIZE = (128, 128)
MODEL_SAVE_PATH = 'model.h5'

def create_model():
    """
    Membuat arsitektur CNN sesuai spesifikasi:
    Conv2D -> ReLU -> MaxPooling
    Conv2D -> ReLU -> MaxPooling
    Flatten
    Dense -> ReLU
    Dense -> Sigmoid
    """
    model = models.Sequential([
        # Layer Konvolusi 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Layer Konvolusi 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten & Fully Connected Layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Output Layer (Binary Classification: Asli vs Palsu)
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def preprocess_image(image_path):
    """
    Preprocessing citra:
    1. Convert ke grayscale
    2. Resize ke 128x128
    3. Normalisasi (0-1)
    """
    img = load_img(image_path, color_mode='grayscale', target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

def train_dummy_model():
    """
    Fungsi untuk melatih model dengan data dummy (hanya untuk demonstrasi aplikasi).
    Dalam implementasi nyata, gunakan dataset tanda tangan asli dan palsu.
    """
    print("Membuat data dummy untuk keperluan demo...")
    # Buat 100 sample data dummy (50 asli, 50 palsu)
    X = np.random.rand(100, 128, 128, 1)
    y = np.array([1]*50 + [0]*50)
    
    model = create_model()
    print("Melatih model (ini mungkin butuh waktu singkat)...")
    model.fit(X, y, epochs=5, batch_size=8)
    
    print(f"Menyimpan model ke {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    # Karena ini adalah aplikasi demo tanpa dataset eksternal yang diunduh, 
    # kita akan membuat model dan menyimpannya langsung.
    train_dummy_model()
