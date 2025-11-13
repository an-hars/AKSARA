import os
import sys

# 1. Ini adalah "kamus" dari file DATA.YML Anda
# Pastikan ini SAMA PERSIS.
MAPPING = [
    'BA', 'CA', 'DA', 'GA', 'HA', 'JA', 'KA', 'LA', 'MA', 'NA', 
    'NGA', 'NYA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA'
]

# 2. Path ke folder label Anda (relatif terhadap tempat skrip ini dijalankan)
LABEL_DIRECTORIES = [
    os.path.join('train', 'labels'),
    os.path.join('valid', 'labels'),
    os.path.join('test', 'labels')
]

def convert_labels():
    """
    Membaca semua file .txt, mengonversinya dari format YOLO
    (id x_center y_center w h) ke format OCR (teks).
    """
    print("Memulai konversi label format YOLO ke OCR...")
    total_files_converted = 0
    
    for dir_path in LABEL_DIRECTORIES:
        if not os.path.exists(dir_path):
            print(f"Warning: Folder '{dir_path}' tidak ditemukan. Melewati...")
            continue
            
        print(f"\nMemproses folder: {dir_path}")
        
        filenames = os.listdir(dir_path)
        if not filenames:
            print("... Folder kosong.")
            continue
            
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(dir_path, filename)
            lines_data = [] # Untuk menyimpan (x_center, class_id)
            
            try:
                # --- Tahap 1: BACA dan Kumpulkan data YOLO ---
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            lines_data.append((x_center, class_id))
                            
                if not lines_data:
                    # print(f"Warning: File {filename} kosong atau formatnya salah.")
                    continue

                # --- Tahap 2: URUTKAN berdasarkan posisi X (kiri ke kanan) ---
                lines_data.sort(key=lambda item: item[0])
                
                # --- Tahap 3: BANGUN string teks OCR ---
                ocr_text = ""
                for x_center, class_id in lines_data:
                    if 0 <= class_id < len(MAPPING):
                        ocr_text += MAPPING[class_id]
                    else:
                        print(f"Error: Class ID {class_id} di {filename} tidak ada di MAPPING!")
                        
                # --- Tahap 4: TIMPA file .txt dengan teks OCR ---
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
                    
                total_files_converted += 1

            except Exception as e:
                print(f"Error saat memproses {filename}: {e}")

    print(f"\n--- Konversi Selesai ---")
    print(f"Total {total_files_converted} file label berhasil dikonversi ke format teks OCR.")

if __name__ == "__main__":
    # Pastikan path sudah benar.
    # Harap jalankan skrip ini dari folder yang berisi folder 'train', 'valid', 'test'.
    
    # Cek cepat
    if not os.path.exists(LABEL_DIRECTORIES[0]):
         print(f"Error: Tidak dapat menemukan folder '{LABEL_DIRECTORIES[0]}'")
         print("Pastikan Anda menjalankan skrip ini dari folder proyek utama Anda.")
         sys.exit(1)
         
    convert_labels()