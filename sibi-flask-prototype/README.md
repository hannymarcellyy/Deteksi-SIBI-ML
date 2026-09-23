# SIBI Bridge - Flask Prototype

Prototype dashboard untuk sistem deteksi bahasa isyarat SIBI berbasis Flask.

## Cara menjalankan

1. Buka folder project:
   ```bash
   cd sibi-flask-prototype
   ```

2. Buat virtual environment:
   ```bash
   python -m venv venv
   ```

3. Aktifkan virtual environment:
   Windows:
   ```bash
   venv\Scripts\activate
   ```
   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

5. Jalankan Flask:
   ```bash
   python app.py
   ```

6. Buka browser:
   ```text
   http://127.0.0.1:5000
   ```

## Catatan

- Hasil translate masih dummy karena model ML belum disambungkan.
- Nanti bagian `DUMMY_TRANSLATIONS` di `app.py` bisa diganti dengan function model prediksi.
