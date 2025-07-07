# =============================================================================
# Nama File: water_quality_analysis.py
# Deskripsi: Script Python ini bertujuan untuk melakukan analisis kualitas air
#            menggunakan teknik Data Mining. Data akan dipreprocessing,
#            kemudian model klasifikasi akan dibangun untuk memprediksi
#            potabilitas air (layak minum atau tidak).
# Tanggal Pengerjaan: 2 Juli 2025
# =============================================================================

# Bagian 1: Import Library yang Dibutuhkan
# Kita import semua library yang akan kita gunakan di awal, biar rapi dan mudah diatur.
# Pandas untuk manipulasi dan analisis data tabular (DataFrame).
# NumPy untuk operasi numerik, seringkali digunakan bersama Pandas.
# Matplotlib dan Seaborn untuk visualisasi data, penting untuk eksplorasi dan presentasi yang menarik.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Library dari Scikit-learn, paket utama untuk Machine Learning di Python.
# train_test_split: Untuk membagi data menjadi set pelatihan dan pengujian.
# StandardScaler: Untuk melakukan penskalaan fitur (feature scaling), penting untuk beberapa algoritma.
# DecisionTreeClassifier: Salah satu algoritma klasifikasi yang akan kita gunakan (Pohon Keputusan).
# LogisticRegression: Algoritma klasifikasi lain yang sederhana namun efektif.
# accuracy_score, precision_score, recall_score, f1_score: Metrik untuk mengevaluasi performa model klasifikasi.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # Tambahan satu lagi, jadi kita punya 2 model!
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =============================================================================
# Bagian 2: Memuat Dataset
# Langkah pertama dalam setiap proyek Data Mining adalah memuat data dari sumbernya.
# Pastikan file 'water_potability.csv' berada di direktori yang sama
# dengan script ini, atau berikan path lengkapnya jika disimpan di lokasi lain.
# =============================================================================
print("--- Memulai Proses Analisis Kualitas Air ---")
try:
    # Menggunakan pd.read_csv untuk membaca file CSV ke dalam objek DataFrame Pandas.
    df = pd.read_csv('water_potability.csv')
    print("Dataset 'water_potability.csv' berhasil dimuat.")
    
    # Menampilkan 5 baris pertama dari DataFrame. Ini membantu kita untuk melihat
    # struktur data, nama kolom, dan contoh nilai-nilai awal.
    print("\n[Preview Data (5 Baris Pertama)]")
    print(df.head())
    
    # Menampilkan informasi ringkas tentang DataFrame. Ini sangat penting untuk
    # mengidentifikasi tipe data setiap kolom, jumlah entri non-null (untuk cek missing values),
    # dan penggunaan memori.
    print("\n[Informasi Umum Dataset]")
    print(df.info())
    
except FileNotFoundError:
    # Menangani kasus jika file dataset tidak ditemukan. Pesan error yang informatif
    # akan ditampilkan agar pengguna tahu apa yang salah.
    print("ERROR: File 'water_potability.csv' tidak ditemukan.")
    print("Pastikan file CSV berada di direktori yang sama dengan script ini.")
    print("Atau, periksa kembali nama file dan path lengkapnya jika sudah dipindahkan.")
    exit() # Hentikan eksekusi script jika file tidak ditemukan.

# =============================================================================
# Bagian 3: Preprocessing Data - Penanganan Missing Values
# Setelah memuat data, langkah selanjutnya adalah membersihkan data (data cleaning).
# Missing values (nilai yang hilang) dapat mempengaruhi performa model secara signifikan.
# Untuk dataset ini, kolom 'ph', 'Sulfate', dan 'Trihalomethanes' teridentifikasi
# memiliki nilai yang hilang. Kita akan mengisi nilai-nilai yang hilang tersebut
# menggunakan metode imputasi dengan nilai rata-rata (mean) dari kolom masing-masing.
# Pendekatan ini umum digunakan untuk data numerik dan relatif sederhana.
# =============================================================================
print("\n--- Tahap Preprocessing Data ---")

# Mengecek dan menampilkan jumlah missing values di setiap kolom sebelum imputasi.
# Ini memberikan gambaran jelas tentang seberapa banyak data yang hilang di setiap fitur.
print("\n[Jumlah Missing Values Sebelum Imputasi]:")
print(df.isnull().sum())

# Melakukan imputasi missing values.
# Fungsi .fillna() digunakan untuk mengganti nilai NaN (Not a Number) dengan nilai yang ditentukan.
# 'df['kolom'].mean()' menghitung rata-rata dari kolom tersebut.
# 'inplace=True' memastikan bahwa perubahan diterapkan langsung pada DataFrame 'df' itu sendiri.
print("\nMelakukan imputasi missing values dengan nilai rata-rata (mean)...")
df['ph'].fillna(df['ph'].mean(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].mean(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)

# Mengecek kembali jumlah missing values setelah proses imputasi
# untuk memverifikasi bahwa semua nilai yang hilang telah berhasil ditangani (seharusnya menjadi 0).
print("\n[Jumlah Missing Values Setelah Imputasi]:")
print(df.isnull().sum())

# Verifikasi ulang informasi dataset setelah imputasi untuk memastikan
# semua kolom 'Non-Null Count' sudah sama dengan total entri (3276).
print("\n[Informasi Dataset Setelah Imputasi Lengkap]:")
print(df.info())

# =============================================================================
# Bagian 4: Pemisahan Fitur (X) dan Target (y)
# Dalam Machine Learning, kita perlu memisahkan data input (fitur)
# dari data output (target atau label).
# Fitur (X) adalah variabel-variabel independen yang akan digunakan untuk
# melakukan prediksi. Target (y) adalah variabel dependen yang ingin kita prediksi.
# Dalam kasus ini, 'Potability' adalah target kita.
# =============================================================================
print("\n--- Memisahkan Fitur dan Target ---")

# Menentukan fitur (X) dengan menghilangkan kolom 'Potability' dari DataFrame.
# axis=1 menunjukkan bahwa kita menghapus kolom, bukan baris.
X = df.drop('Potability', axis=1) 
# Menentukan target (y) sebagai kolom 'Potability'.
y = df['Potability']

print(f"Dimensi Fitur (X) setelah pemisahan: {X.shape}") # Menampilkan bentuk (baris, kolom) dari fitur
print(f"Dimensi Target (y) setelah pemisahan: {y.shape}") # Menampilkan bentuk (baris) dari target

# =============================================================================
# Bagian 5: Pembagian Data Training dan Testing
# Dataset perlu dibagi menjadi set pelatihan (training set) dan set pengujian (testing set).
# Training set digunakan untuk melatih model, sedangkan testing set digunakan
# untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.
# Ini penting untuk menilai kemampuan generalisasi model dan menghindari overfitting.
# 'test_size=0.2' berarti 20% data akan menjadi testing set, dan 80% sisanya training set.
# 'random_state=42' digunakan untuk memastikan pembagian data yang konsisten
# setiap kali script dijalankan, yang penting untuk reproduksibilitas hasil.
# =============================================================================
print("\n--- Membagi Data menjadi Training dan Testing Set ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Ukuran X_train (Fitur Pelatihan): {X_train.shape}")
print(f"Ukuran X_test (Fitur Pengujian): {X_test.shape}")
print(f"Ukuran y_train (Target Pelatihan): {y_train.shape}")
print(f"Ukuran y_test (Target Pengujian): {y_test.shape}")

# =============================================================================
# Bagian 6: Feature Scaling (Penskalaan Fitur)
# Penskalaan fitur adalah proses mengubah skala fitur numerik agar berada
# dalam rentang nilai yang seragam. Ini penting untuk banyak algoritma
# Machine Learning (misalnya, yang berbasis jarak seperti SVM atau K-NN,
# atau yang menggunakan gradien seperti Regresi Logistik) karena dapat
# mencegah fitur dengan nilai yang lebih besar mendominasi perhitungan.
# Kita akan menggunakan StandardScaler, yang menstandarisasi fitur
# dengan menghilangkan rata-rata dan menskalakan ke varians unit.
# =============================================================================
print("\n--- Melakukan Feature Scaling (Penskalaan Fitur) ---")

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Melatih scaler pada data training (X_train) dan kemudian mentransformasi
# baik data training maupun data testing. Penting untuk hanya melatih scaler
# pada data training untuk menghindari data leakage dari testing set.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Fitur telah berhasil diskalakan menggunakan StandardScaler.")
print(f"Contoh data X_train_scaled (5 baris pertama):\n{X_train_scaled[:5]}")


# =============================================================================
# Bagian 7: Implementasi Model Klasifikasi (Minimal 2 Teknik)
# Sesuai instruksi, kita akan mengimplementasikan minimal dua teknik klasifikasi:
# 1. Decision Tree Classifier (Pohon Keputusan)
# 2. Logistic Regression (Regresi Logistik)
# =============================================================================
print("\n--- Membangun dan Melatih Model Klasifikasi ---")

# --- Model 1: Decision Tree Classifier ---
print("\n[Model 1: Decision Tree Classifier]")
dt_classifier = DecisionTreeClassifier(random_state=42) # Inisialisasi model Decision Tree
dt_classifier.fit(X_train_scaled, y_train) # Melatih model menggunakan data training yang sudah diskalakan
y_pred_dt = dt_classifier.predict(X_test_scaled) # Melakukan prediksi pada data testing

# --- Model 2: Logistic Regression ---
print("\n[Model 2: Logistic Regression]")
lr_classifier = LogisticRegression(random_state=42, solver='liblinear') # Inisialisasi model Logistic Regression
# 'solver' dispesifikasikan untuk menghindari warning, 'liblinear' cocok untuk dataset kecil-menengah
lr_classifier.fit(X_train_scaled, y_train) # Melatih model
y_pred_lr = lr_classifier.predict(X_test_scaled) # Melakukan prediksi

print("Model-model klasifikasi berhasil dilatih.")


# =============================================================================
# Bagian 8: Evaluasi Hasil Model
# Setelah model dilatih, penting untuk mengevaluasi performanya
# menggunakan metrik yang sesuai. Untuk masalah klasifikasi, metrik umum meliputi:
# - Akurasi (Accuracy): Proporsi prediksi yang benar dari total prediksi.
# - Presisi (Precision): Proporsi true positive dari total prediksi positif (penting untuk false positive).
# - Recall: Proporsi true positive dari total aktual positif (penting untuk false negative).
# - F1-Score: Rata-rata harmonik dari Precision dan Recall.
# - Confusion Matrix: Tabel yang menunjukkan jumlah true/false positives/negatives.
# =============================================================================
print("\n--- Mengevaluasi Performa Model ---")

# --- Evaluasi Model 1: Decision Tree Classifier ---
print("\n[Evaluasi Model Decision Tree Classifier]")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# --- Evaluasi Model 2: Logistic Regression ---
print("\n[Evaluasi Model Logistic Regression]")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# =============================================================================
# Bagian 9: Visualisasi Hasil (Opsional tapi direkomendasikan untuk presentasi)
# Visualisasi dapat membantu dalam memahami performa model dan distribusi data.
# Ini bisa jadi nilai tambah besar untuk presentasi ke dosen.
# =============================================================================
print("\n--- Melakukan Visualisasi Hasil (Contoh Sederhana) ---")

# Visualisasi Confusion Matrix untuk Decision Tree
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) # 1 baris, 2 kolom, plot pertama
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.title('Confusion Matrix: Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Visualisasi Confusion Matrix untuk Logistic Regression
plt.subplot(1, 2, 2) # 1 baris, 2 kolom, plot kedua
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout() # Merapikan layout antar subplot
plt.show() # Menampilkan plot

print("\nAnalisis Data Mining telah selesai!")
print("--- SCRIPT SELESAI ---")