import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
from PIL import Image
import os
import math

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
CORS(app) 

@app.route('/')
def halaman_utama(): 
    """Halaman utama untuk mengecek apakah server berjalan."""
    print("GET request ke / diterima.")
    return "Halo! Server OCR Flask sudah berjalan. Siap menerima POST di /predict."

# --- Variabel Global ---
prediction_model = None
char_to_num = None
num_to_char = None
IMG_WIDTH = 512
IMG_HEIGHT = 64

# --- 2. Fungsi Helper (Penting) ---

def build_prediction_model(img_width, img_height, num_classes):
    """
    Membangun arsitektur model HANYA untuk prediksi.
    Ini harus memiliki nama layer yang SAMA PERSIS dengan model training.
    """
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image")
    # Bagian CNN
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2d")(input_img)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2d_1")(x)
    x = layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    # Reshape
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    # Dense + Dropout
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    # Bagian RNN
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="bidirectional")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="bidirectional_1")(x)
    # Bagian Output
    x = layers.Dense(num_classes + 1, activation="softmax", name="output")(x)
    model = keras.Model(inputs=input_img, outputs=x, name="OCR_Prediction_Model")
    return model

def load_vocab(path="vocab.txt"):
    """
    Memuat vocabulary dari file vocab.txt.
    """
    if not os.path.exists(path):
        print(f"Error: File '{path}' tidak ditemukan.")
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        vocab_from_file = [line.strip() for line in f.readlines() if line.strip()]
    
    if not vocab_from_file:
        print(f"Error: File '{path}' kosong.")
        return None, None
    
    print(f"Vocabulary dari file dimuat ({len(vocab_from_file)} karakter).")
    
    char_to_num_layer = layers.StringLookup(vocabulary=vocab_from_file, mask_token=None, oov_token="[UNK]")
    full_vocab = char_to_num_layer.get_vocabulary()
    num_to_char_layer = layers.StringLookup(
        vocabulary=full_vocab, mask_token=None, oov_token="[UNK]", invert=True
    )
    print(f"Total kelas (termasuk [UNK]): {len(full_vocab)}")
    return char_to_num_layer, num_to_char_layer, len(full_vocab)

# (Fungsi preprocess_image dan ctc_decoder tidak berubah)
def preprocess_image(img_data):
    img = img_data.convert('L')
    w, h = img.size
    target_w, target_h = IMG_WIDTH, IMG_HEIGHT
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    new_img = Image.new('L', (target_w, target_h), 0) 
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    img_array = np.array(new_img)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_array = img_array / 255.0
    img_array = 1.0 - img_array
    img_array = tf.transpose(img_array, perm=[1, 0])
    img_array = tf.expand_dims(img_array, axis=-1)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def ctc_decoder(pred, num_to_char_layer):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=10)[0][0]
    text_results = []
    for res in results:
        res_numpy = res.numpy()
        valid_indices = [i for i in res_numpy if i != -1]
        text_tensor = num_to_char_layer(tf.convert_to_tensor(valid_indices, dtype=tf.int32))
        text = tf.strings.reduce_join(text_tensor).numpy().decode("utf-8")
        text_results.append(text)
    return text_results[0]


# --- 3. Fungsi untuk Memuat Model (PERBAIKAN FINAL) ---
def load_app_model():
    global prediction_model, char_to_num, num_to_char
    
    print("Memuat vocabulary...")
    char_to_num, num_to_char, num_classes = load_vocab("vocab.txt")
    if char_to_num is None:
        print("Gagal memuat vocabulary. Server tidak bisa memulai.")
        return
    
    print(f"Membangun arsitektur PREDIKSI dengan {num_classes} kelas...")
    
    try:
        prediction_model = build_prediction_model(IMG_WIDTH, IMG_HEIGHT, num_classes)
        print("Arsitektur prediksi berhasil dibangun. (Output units: 14)")
    except Exception as e:
        print(f"Error saat membangun arsitektur: {e}")
        return

    model_path = "model_crnn_terbaik.keras"
    if not os.path.exists(model_path):
        print(f"Error: File model '{model_path}' tidak ditemukan.")
        return
        
    print(f"Memuat weights dari {model_path}...")
    try:
        # --- INI DIA PERBAIKANNYA ---
        # Kita hapus .expect_partial() yang salah
        prediction_model.load_weights(model_path)
        # ----------------------------
        
        print("Weights berhasil dimuat ke model prediksi.")
        
        prediction_model.summary()
        print("Model prediksi berhasil dibuat.")
        
        dummy_input = tf.zeros((1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32)
        prediction_model(dummy_input)
        print("Model warm-up selesai.")
        
    except Exception as e:
        print(f"Error saat memuat weights: {e}")
        prediction_model = None

# --- 4. API Endpoint untuk Prediksi (Tidak berubah) ---
@app.route('/predict', methods=['POST'])
def predict():
    if not prediction_model or not num_to_char:
        print("Error: Model belum siap saat request /predict diterima.")
        return jsonify({'error': 'Model belum siap atau gagal dimuat'}), 500
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'Key "image" tidak ditemukan di JSON body'}), 400
    try:
        img_data_b64 = data['image'].split(',')[1]
        img_data_bytes = base64.b64decode(img_data_b64)
        img = Image.open(io.BytesIO(img_data_bytes))
        processed_img = preprocess_image(img)
        raw_prediction = prediction_model.predict(processed_img)
        text_result = ctc_decoder(raw_prediction, num_to_char)
        print(f"Prediksi: {text_result}")
        return jsonify({'prediction': text_result})
    except Exception as e:
        print(f"Error pada endpoint /predict: {e}")
        return jsonify({'error': str(e)}), 400

# --- 5. Jalankan Server ---
if __name__ == '__main__':
    load_app_model()
    print("Menjalankan server Flask...")
    app.run(host='0.0.0.0', port=5000, debug=False)