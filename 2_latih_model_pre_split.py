import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
import math # <-- PERBAIKAN 1: Tambahkan import math

# --- Variabel Konfigurasi ---
IMG_HEIGHT = 64
IMG_WIDTH = 512
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Path ke folder dataset Anda
TRAIN_PATH = 'train'
VALID_PATH = 'valid'
TEST_PATH = 'test'


# --- 1. Fungsi Memuat Data (BARU) ---
def load_dataset_from_split(split_path):
    
    images_dir = os.path.join(split_path, 'images')
    labels_dir = os.path.join(split_path, 'labels')
    
    data = []
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Folder 'images' atau 'labels' tidak ditemukan di {split_path}")
        return data

    print(f"Memuat data dari {split_path}...")
    
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
                
                if label:
                    data.append((img_path, label))
            except Exception as e:
                print(f"Warning: Gagal membaca label {label_path}: {e}")
        else:
            print(f"Warning: Label file tidak ditemukan untuk {img_name}")
            
    return data

# --- 2. Fungsi Pre-processing ---

def preprocess_image(image_path, img_height, img_width):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1) # Grayscale
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, img_height, img_width)
        image = 1.0 - image # Invert warna
        image = tf.transpose(image, perm=[1, 0, 2]) # (H, W, C) -> (W, H, C)
        return image
    except Exception as e:
        print(f"Error memproses gambar {image_path}: {e}")
        return None

def vectorize_label(label, char_to_num_map):
    try:
        label_indices = [char_to_num_map[char] for char in label]
        return tf.convert_to_tensor(label_indices, dtype=tf.int32)
    except KeyError as e:
        print(f"Error: Karakter '{e.args[0]}' dalam label '{label}' tidak ada di CHAR_SET.")
        return None
    except Exception as e:
        print(f"Error vektorisasi label {label}: {e}")
        return None

# Fungsi untuk membuat tf.data.Dataset
def create_tf_dataset(data_pairs, char_to_num_map, batch_size, augment=False):
    image_paths = [item[0] for item in data_pairs]
    labels = [item[1] for item in data_pairs]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    def map_fn(path, lbl):
        image, label_vec = tf.py_function(
            func=lambda p, l: (
                preprocess_image(p.numpy().decode('utf-8'), IMG_HEIGHT, IMG_WIDTH),
                vectorize_label(l.numpy().decode('utf-8'), char_to_num_map)
            ),
            inp=[path, lbl],
            Tout=[tf.float32, tf.int32]
        )
        image.set_shape([IMG_WIDTH, IMG_HEIGHT, 1])
        label_vec.set_shape([None])
        return {"image": image, "label": label_vec}

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x: x["image"] is not None and x["label"] is not None)
    
    if augment:
        pass 
        
    return (
        dataset
        .padded_batch(batch_size, 
                       padded_shapes={"image": [IMG_WIDTH, IMG_HEIGHT, 1], "label": [None]},
                       padding_values={"image": 0.0, "label": -1}) 
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

# --- 3. Arsitektur Model dan CTC Loss (Sudah Benar) ---

class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        y_true_unpadded = tf.ragged.boolean_mask(y_true, tf.not_equal(y_true, -1)).to_tensor(default_value=0)

        loss = self.loss_fn(y_true_unpadded, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

    def get_config(self):
        config = super().get_config()
        return config

def build_crnn_model(img_width, img_height, num_classes):
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="int32")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(num_classes + 1, activation="softmax", name="output")(x)
    output = CTCLayer(name="ctc_loss")(labels, x)

    model = keras.Model(inputs=[input_img, labels], outputs=output, name="OCR_CRNN_Model")
    
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    return model

# --- 4. Fungsi Utama (Main) ---

def main():
    # 1. Muat semua data dari folder
    train_data = load_dataset_from_split(TRAIN_PATH)
    valid_data = load_dataset_from_split(VALID_PATH)
    test_data = load_dataset_from_split(TEST_PATH)

    if not train_data or not valid_data:
        print("Error: Data training atau validasi kosong. Harap cek path dan file label Anda.")
        return

    # --- PERBAIKAN 2: Hitung steps ---
    # Lakukan ini SEBELUM membuat dataset
    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
    validation_steps = math.ceil(len(valid_data) / BATCH_SIZE)
    
    print(f"Total data train: {len(train_data)} -> Steps per epoch: {steps_per_epoch}")
    print(f"Total data valid: {len(valid_data)} -> Validation steps: {validation_steps}")
    # ---------------------------------

    # 2. Buat CHAR_SET secara dinamis
    all_labels = [label for _, label in train_data + valid_data + test_data]
    all_chars = "".join(all_labels)
    char_set = sorted(list(set(all_chars)))
    
    if ' ' not in char_set:
        char_set.append(' ')
        
    CHAR_SET = "".join(char_set)
    print(f"Karakter Ditemukan ({len(CHAR_SET)}): {CHAR_SET}")

    # 3. Buat mapping karakter
    char_to_num = layers.StringLookup(vocabulary=list(CHAR_SET), mask_token=None)
    
    vocab = char_to_num.get_vocabulary()
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for char in vocab:
            f.write(f"{char}\n")
    print("Vocabulary disimpan ke vocab.txt")

    char_to_num_map = {char: i for i, char in enumerate(vocab)}

    # 4. Buat tf.data.Dataset
    train_dataset = create_tf_dataset(train_data, char_to_num_map, BATCH_SIZE, augment=True)
    valid_dataset = create_tf_dataset(valid_data, char_to_num_map, BATCH_SIZE)
    test_dataset = create_tf_dataset(test_data, char_to_num_map, BATCH_SIZE)
    
    # --- PERBAIKAN 3: Tambahkan .repeat() ---
    train_dataset = train_dataset.repeat()
    valid_dataset = valid_dataset.repeat()
    # ---------------------------------------
    
    print("Dataset pipeline berhasil dibuat.")

    # 5. Bangun dan latih model
    num_classes = len(vocab)
    model = build_crnn_model(IMG_WIDTH, IMG_HEIGHT, num_classes)
    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "model_crnn_terbaik.keras", monitor="val_loss", save_best_only=True, save_weights_only=False
    )

    print("Memulai pelatihan model...")
    # --- PERBAIKAN 4: Tambahkan steps ke .fit() ---
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    # ---------------------------------------------

    print("Pelatihan selesai.")

    # 6. Evaluasi model
    if test_data:
        print("Mengevaluasi model pada test set...")
        best_model = keras.models.load_model(
            "model_crnn_terbaik.keras", 
            custom_objects={"CTCLayer": CTCLayer}
        )
        
        # Hitung juga test steps jika perlu
        test_steps = math.ceil(len(test_data) / BATCH_SIZE)
        print(f"Total data test: {len(test_data)} -> Test steps: {test_steps}")
        
        test_loss = best_model.evaluate(test_dataset, steps=test_steps)
        print(f"Final Test Loss: {test_loss}")
    else:
        print("Tidak ada test data untuk evaluasi.")

if __name__ == "__main__":
    main()