# Deteksi Keaslian Tanda Tangan Menggunakan Convolutional Neural Network Berbasis Citra Digital

Aplikasi web berbasis Python untuk mendeteksi keaslian tanda tangan menggunakan metode Deep Learning **Convolutional Neural Network (CNN)**. Sistem ini dapat mengklasifikasikan citra tanda tangan menjadi dua kategori: **Genuine (Asli)** atau **Forged (Palsu)**.

![Premium UI Preview](https://img.shields.io/badge/UI-Premium-blue)
![Backend](https://img.shields.io/badge/Backend-Flask-green)
![ML](https://img.shields.io/badge/ML-TensorFlow-orange)

## 🚀 Fitur Utama

- **Real-time Prediction**: Klasifikasi cepat menggunakan model CNN yang sudah terlatih.
- **Modern UI**: Antarmuka berbasis web dengan desain premium (Glassmorphism), responsif, dan animasi halus.
- **Drag & Drop**: Memudahkan user untuk mengunggah gambar tanda tangan.
- **Preprocessing Otomatis**: Konversi ke grayscale, resizing, dan normalisasi dilakukan di sisi server sebelum prediksi.

## 🛠️ Teknologi yang Digunakan

### Backend
- **Python**: Bahasa pemrograman utama.
- **Flask**: Framework web minimalis untuk server.
- **TensorFlow / Keras**: Framework Deep Learning untuk model CNN.
- **NumPy & OpenCV**: Pengolahan array dan citra digital.

### Frontend
- **HTML5 & Vanilla CSS**: Struktur dan desain custom premium.
- **Bootstrap 5**: Keperluan layouting dan komponen UI.
- **JavaScript (ES6)**: Menangani logika upload secara asinkron (AJAX/Fetch).
- **Font Awesome**: Koleksi icon modern.

## 🏗️ Arsitektur Model CNN

Model dibangun dengan urutan layer berikut:
1. **Conv2D**: Ekstraksi fitur spasial dengan ReLU.
2. **MaxPooling2D**: Reduksi dimensi citra.
3. **Conv2D**: Pendalaman ekstraksi fitur tingkat lanjut.
4. **Flatten**: Konversi fitur multidimensi menjadi vektor tunggal.
5. **Dense (128 units)**: Fully connected layer untuk pemrosesan informasi.
6. **Dense (1 unit - Sigmoid)**: Output biner (0-1) untuk probabilitas klasifikasi.

## 💻 Cara Instalasi & Menjalankan

### 1. Prasyarat
Pastikan Anda sudah menginstal Python 3.10+ dan library yang dibutuhkan:
```bash
pip install flask tensorflow numpy pillow
```

### 2. Jalankan Aplikasi
Anda bisa langsung menjalankan file batch yang tersedia:
```bash
start.bat
```
*Atau jalankan secara manual:*
```bash
python train_model.py  # Jika model.h5 belum ada
python app.py
```

### 3. Akses Web
Buka browser Anda dan akses:
`http://127.0.0.1:5000`

## 📁 Struktur Folder
```text
artikel/
├── static/
│   ├── uploads/    # Folder penyimpanan gambar sementara
│   └── style.css   # Custom CSS Styling
├── templates/
│   └── index.html  # Halaman UI Utama
├── app.py          # Backend (Flask)
├── train_model.py  # Script pelatihan model
├── start.bat       # Shortcut menjalankan aplikasi
└── README.md       # Dokumentasi proyek
```

## 📝 Catatan Penting
Model yang disertakan dalam repositori ini dilatih menggunakan **Dummy Data** untuk keperluan demonstrasi fungsionalitas aplikasi. Untuk penggunaan produksi, silakan latih ulang `train_model.py` menggunakan dataset tanda tangan asli (seperti dataset ICDAR atau SigComp).

---
**Dibuat untuk keperluan akademik & riset AI.**
