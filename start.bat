@echo off
echo ==========================================
echo   Signature Authenticity Detection App
echo ==========================================
echo.

:: Cek apakah model sudah ada, jika belum jalankan training
if not exist "model.h5" (
    echo [INFO] Model tidak ditemukan. Menjalankan training pertama kali...
    python train_model.py
)

echo [INFO] Menyalakan server Flask...
echo [TIPS] Buka http://127.0.0.1:5000 di browser Anda.
echo.

python app.py

pause
