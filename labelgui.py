import csv
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Konfigurasi ---
# PERBAIKI INI SESUAI SISTEM ANDA (contoh di Windows)
# IMAGE_FOLDER = 'D:/Action Lomba/data-mining-action-2025/train/train' 
IMAGE_FOLDER = r'C:\Users\adief\OneDrive\Dokumen\Lomba\Action UNESA 2025\train\train' # Contoh path relatif yang mungkin benar jika folder ada di samping .py

# CEK INI SESUAI EKSTENSI FILE ANDA
FILE_EXTENSION = '.jpg' # Ubah ke '.jpg' jika file gambar Anda berekstensi .jpg

INPUT_CSV = r'C:\Users\adief\OneDrive\Dokumen\Lomba\Action UNESA 2025\train\train_labels.csv'
OUTPUT_CSV = r'C:\Users\adief\OneDrive\Dokumen\Lomba\Action UNESA 2025\train\hasil_label_gui.csv' 
FILENAME_COLUMN_INDEX = 0         

# --- PERUBAHAN INI UNTUK RESOLUSI GAMBAR ---
MAX_IMAGE_WIDTH = 800             # Naikkan resolusi gambar (dari 600 ke 800)
# Anda bisa coba 1000 atau 1200 jika layar Anda besar, tapi hati-hati agar tidak terlalu besar

# --- PERUBAHAN INI UNTUK DAFTAR LABEL BARU ---
LABELS_CHOICES = [
    "Ayam Bakar",
    "Ayam Betutu",
    "Ayam Goreng",
    "Ayam Pop",
    "Bakso",
    "Coto Makassar",
    "Gado Gado",
    "Gudeg",
    "Nasi Goreng",
    "Pempek",
    "Rawon",
    "Rendang",
    "Sate Madura",
    "Sate Padang",
    "Soto",
    "Kotak Putih"
]
# --------------------

class ImageLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Labeling Gambar (Bisa Dilanjutkan)")

        self.all_input_rows = []       
        self.rows_to_process = []      
        self.data_untuk_ditulis = []   
        self.header = []
        self.current_index = 0         
        
        self.label_column_index = -1   

        self.setup_ui()

        self.load_csv_and_progress()
        if self.rows_to_process: 
            self.load_next_image() 
        else:
            self.root.quit()

    def setup_ui(self):
        self.filename_label = tk.Label(self.root, text="Memuat...", font=("Arial", 12))
        self.filename_label.pack(pady=10)
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10, padx=20)
        
        # --- PERUBAHAN DI SINI: Frame untuk menampung tombol-tombol label ---
        self.labels_frame = tk.Frame(self.root)
        self.labels_frame.pack(pady=10)

        # Buat tombol secara dinamis dari daftar LABELS_CHOICES
        # Kita akan tata dalam beberapa baris jika terlalu banyak
        column_count = 3 # Jumlah kolom tombol
        for i, label_text in enumerate(LABELS_CHOICES):
            button = tk.Button(self.labels_frame, text=label_text, 
                               font=("Arial", 12), padx=10, pady=5,
                               command=lambda text=label_text: self.save_choice(text))
            row_num = i // column_count
            col_num = i % column_count
            button.grid(row=row_num, column=col_num, padx=5, pady=5, sticky="ew") # sticky="ew" agar tombol melebar

        # Konfigurasi kolom agar melebar
        for col in range(column_count):
            self.labels_frame.grid_columnconfigure(col, weight=1)
        # --- AKHIR PERUBAHAN UI ---
        
        self.root.protocol("WM_DELETE_WINDOW", self.save_on_close)

    def load_csv_and_progress(self):
        try:
            with open(INPUT_CSV, mode='r', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                self.header = next(reader) 
                self.all_input_rows = list(reader)
            if not self.all_input_rows:
                messagebox.showerror("Error", f"File input '{INPUT_CSV}' kosong.")
                return
        except FileNotFoundError:
            messagebox.showerror("Error", f"File INPUT '{INPUT_CSV}' tidak ditemukan.")
            self.root.quit()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca '{INPUT_CSV}': {e}")
            self.root.quit()
            return

        try:
            self.label_column_index = self.header.index('Label')
            print(f"Kolom 'Label' ditemukan di index {self.label_column_index}.")
        except ValueError:
            self.label_column_index = len(self.header)
            self.header.append('Label')
            print(f"Kolom 'Label' tidak ditemukan, menambahkannya di index {self.label_column_index}.")
        
        self.data_untuk_ditulis = [self.header] 

        labeled_filenames = set()
        if os.path.exists(OUTPUT_CSV):
            try:
                with open(OUTPUT_CSV, mode='r', encoding='utf-8') as f_out:
                    reader = csv.reader(f_out)
                    output_header = next(reader) 
                    
                    if output_header != self.header:
                         messagebox.showwarning("Warning", "Header file output tidak cocok! Memulai dari awal.")
                         self.data_untuk_ditulis = [self.header]
                    else:
                        for row in reader:
                            if row: 
                                self.data_untuk_ditulis.append(row)
                                if len(row) > self.label_column_index and row[self.label_column_index]:
                                    labeled_filenames.add(row[FILENAME_COLUMN_INDEX])
                
                if labeled_filenames:
                    messagebox.showinfo("Melanjutkan Pekerjaan", 
                                        f"Ditemukan {len(labeled_filenames)} gambar yang sudah dilabel.\n"
                                        "Melanjutkan sisa pekerjaan Anda.")
            except Exception as e:
                messagebox.showwarning("Warning", f"Gagal membaca file output '{OUTPUT_CSV}': {e}\nMemulai dari awal.")
                self.data_untuk_ditulis = [self.header] 

        self.rows_to_process = []
        for row in self.all_input_rows:
            filename = row[FILENAME_COLUMN_INDEX]
            if filename not in labeled_filenames:
                self.rows_to_process.append(row)

        if not self.rows_to_process:
            if labeled_filenames:
                 messagebox.showinfo("Selesai", "Semua gambar di file input sudah terdeteksi selesai dilabel!")
            else:
                messagebox.showerror("Error", "File CSV input tidak memiliki data.")
            

    def load_next_image(self):
        while self.current_index < len(self.rows_to_process):
            
            self.current_row_data = self.rows_to_process[self.current_index]
            base_filename = self.current_row_data[FILENAME_COLUMN_INDEX]
            
            image_filename = f"{base_filename}{FILE_EXTENSION}"
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            
            total_sisa = len(self.rows_to_process) - self.current_index
            self.filename_label.config(text=f"Mencoba: {image_filename} ({self.current_index + 1} / {len(self.rows_to_process)})")
            self.root.update_idletasks() 

            try:
                img = Image.open(image_path)
                width, height = img.size
                if width > MAX_IMAGE_WIDTH:
                    ratio = MAX_IMAGE_WIDTH / width
                    new_height = int(height * ratio)
                    img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)

                tk_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=tk_image)
                self.image_label.image = tk_image

                self.filename_label.config(text=f"{image_filename} ({self.current_index + 1} / {len(self.rows_to_process)} | Sisa: {total_sisa})")
                return 

            except FileNotFoundError:
                print(f"File {image_path} tidak ditemukan. Melompat...")
                self._save_error_and_skip('ERROR_FILE_NOT_FOUND') 

            except Exception as e:
                print(f"!! ERROR GAGAL MEMUAT {image_path}: {e}")
                print("   File ini kemungkinan rusak (corrupt). Melompat...")
                self._save_error_and_skip(f'ERROR_CORRUPT_OR_LOAD_FAILED')

        self.finish_labeling()
        
    def _save_error_and_skip(self, label):
        self._write_label_to_row(label)
        self.current_index += 1 

    def save_choice(self, label):
        self._write_label_to_row(label)
        self.current_index += 1
        self.load_next_image()

    def _write_label_to_row(self, label):
        labeled_row = list(self.current_row_data) 
        
        while len(labeled_row) <= self.label_column_index:
            labeled_row.append('')
            
        labeled_row[self.label_column_index] = label
        
        self.data_untuk_ditulis.append(labeled_row)
        print(f"Menyimpan: {self.current_row_data[FILENAME_COLUMN_INDEX]} -> {label} (di kolom {self.label_column_index})")


    def finish_labeling(self):
        self.save_to_csv()
        messagebox.showinfo("Selesai!", f"Semua gambar selesai dilabel.\nData disimpan di {OUTPUT_CSV}")
        self.root.quit()

    def save_on_close(self):
        if messagebox.askyesno("Tutup", "Anda yakin ingin keluar? \nHasil yang sudah dilabel akan disimpan."):
            self.save_to_csv()
            self.root.destroy()
            
    def save_to_csv(self):
        try:
            with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.data_untuk_ditulis)
            print(f"Data berhasil disimpan (lengkap) ke {OUTPUT_CSV}")
        except PermissionError:
             messagebox.showerror("Error Simpan", f"Gagal menyimpan ke '{OUTPUT_CSV}'.\n"
                                  "Pastikan file tersebut tidak sedang dibuka di Excel.")
        except Exception as e:
            print(f"Error saat menyimpan CSV: {e}")

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelerApp(root)
    if app.rows_to_process or not os.path.exists(OUTPUT_CSV):
        root.mainloop()
    else:
        print("Tidak ada gambar baru untuk dilabel.")