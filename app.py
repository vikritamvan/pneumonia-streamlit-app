import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image # Untuk membuka dan memproses gambar
import numpy as np
import os # Untuk memeriksa keberadaan file model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Pneumonia", # Judul yang muncul di tab browser
    page_icon="🩺", # Ikon di tab browser
    layout="centered", # Tata letak konten utama: "centered" atau "wide"
    initial_sidebar_state="auto" # Status sidebar saat pertama kali dibuka
)

# --- Fungsi untuk Memuat Model (dengan caching) ---
# Menggunakan st.cache_resource untuk memuat model hanya sekali
# Ini penting karena model besar tidak perlu dimuat ulang setiap kali ada interaksi UI
@st.cache_resource
def load_pneumonia_model():
    model_path = "pneumonia_detection_model.keras" # Nama file model Anda
    
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}. Pastikan model berada di direktori yang sama dengan 'app.py'.")
        st.stop() # Hentikan aplikasi jika model tidak ditemukan

    # Memuat model Keras. custom_objects diperlukan untuk metrik F1Score
    try:
        model = keras.models.load_model(
            model_path,
            # Pastikan definisi F1Score di Python sama persis dengan yang digunakan saat training
            custom_objects={'f1_score': keras.metrics.F1Score(threshold=0.5)}
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop() # Hentikan aplikasi jika model gagal dimuat

# Muat model saat aplikasi dimulai (ini akan dilakukan hanya sekali berkat @st.cache_resource)
model = load_pneumonia_model()

# --- Fungsi Pra-pemrosesan Gambar ---
# Fungsi ini harus meniru pra-pemrosesan yang Anda lakukan di Python sebelum training.
# Lapisan augmentasi TIDAK PERLU diterapkan saat prediksi.
# Hanya lapisan Rescaling (jika ada) dan konversi format yang perlu.
def preprocess_image(image_file, target_size=(224, 224)):
    # Buka gambar menggunakan PIL (Pillow)
    img = Image.open(image_file)
    
    # Konversi ke grayscale jika bukan (X-ray seringkali grayscale)
    # Jika mode adalah 'L' (grayscale) atau 'LA' (grayscale + alpha), biarkan.
    # Jika RGB/RGBA, konversi ke grayscale dulu.
    if img.mode not in ('L', 'LA'):
        img = img.convert('L') # Convert to grayscale

    # Ubah ukuran gambar menggunakan algoritma Lanczos untuk kualitas tinggi
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # Konversi gambar PIL ke array NumPy dengan tipe data float32
    img_array = np.array(img, dtype=np.float32)

    # Jika gambar memiliki 1 channel (grayscale), duplikasi menjadi 3 channel (RGB)
    # Model Anda mengharapkan input (224, 224, 3)
    if img_array.ndim == 2: # Jika array hanya (height, width)
        img_array = np.stack([img_array, img_array, img_array], axis=-1) # Menjadi (height, width, 3)
    elif img_array.shape[-1] == 4: # Jika ada alpha channel (RGBA), buang alpha
        img_array = img_array[:, :, :3]
    
    # Normalisasi: Model Anda memiliki Rescaling(1./127.5, offset=-1) sebagai layer pertama setelah input.
    # Ini berarti gambar input diharapkan dalam rentang [0, 255] dan model akan menormalisasinya sendiri.
    # Jadi, kita tidak perlu normalisasi manual di sini jika img_array sudah di [0, 255].
    # PIL Image.open() dan np.array() akan menghasilkan nilai 0-255 untuk gambar standar.

    # Tambahkan dimensi batch (untuk satu gambar)
    img_array = np.expand_dims(img_array, axis=0) # Menjadi (1, height, width, channels)

    return img_array

# --- Sidebar untuk Navigasi Halaman ---
st.sidebar.title("Navigasi")
# Membuat radio button di sidebar untuk memilih halaman
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi"])

# --- Konten Halaman Berdasarkan Pilihan di Sidebar ---

if page == "Beranda":
    # Menggunakan HTML mentah untuk mengatur judul utama ke tengah
    st.markdown(f'<div style="text-align: center;"><h1><b>Sistem Prediksi Pneumonia dari Citra X-Ray</b></h1></div>', unsafe_allow_html=True)
    
    # Judul Inventor dengan garis bawah dan di tengah
    st.markdown("<h3 style='text-align: center;'><u>Inventor :</u></h3>", unsafe_allow_html=True)
    
    # Daftar Inventor tanpa bullet points, di tengah
    st.markdown("<p style='text-align: center;'>Puspita Kartikasari, S.Si., M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Prof. Dr. Rukun Santoso, M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Dra. Suparti, M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rizwan Arisandi, S.Si., M.Si.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Vikri Haikal</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # Baris kosong untuk pemisah
    
    # Detail Universitas dan Tahun, di tengah
    st.markdown("<h3 style='text-align: center;'>DEPARTEMEN STATISTIKA</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>FAKULTAS SAINS DAN MATEMATIKA</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>UNIVERSITAS DIPONEGORO</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>TAHUN 2025</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # Baris kosong
    
    # Deskripsi aplikasi
    st.write("Aplikasi ini menggunakan model Deep Learning yang dilatih dengan teknik Transfer Learning untuk memprediksi apakah citra X-Ray paru-paru menunjukkan tanda-tanda Pneumonia atau Normal.")
    st.write("Navigasikan ke halaman 'Prediksi' di sidebar untuk mengunggah gambar dan mendapatkan hasilnya.")

elif page == "Prediksi":
    st.title("Prediksi Pneumonia")

    st.write("Unggah gambar X-Ray paru-paru (format JPG/PNG) untuk mendapatkan prediksi.")
    # Widget untuk mengunggah file
    uploaded_file = st.file_uploader("Unggah Gambar X-Ray Paru-paru", type=["jpg", "jpeg", "png"])

    # Hanya tampilkan bagian ini jika ada file yang diunggah
    if uploaded_file is not None:
        st.subheader("Gambar yang Diunggah:")
        # Menampilkan gambar yang diunggah
        st.image(uploaded_file, caption='Gambar X-Ray Anda', use_column_width=True)

        # Tombol untuk memicu prediksi
        if st.button("Lakukan Prediksi"):
            # Menampilkan spinner loading
            with st.spinner("Melakukan prediksi..."):
                try:
                    # Pra-pemrosesan gambar
                    img_for_pred = preprocess_image(uploaded_file)
                    
                    # Melakukan prediksi menggunakan model
                    # Output dari model.predict() adalah array, ambil nilai skalarnya
                    prediction = model.predict(img_for_pred)[0][0] 
                    
                    # Interpretasi hasil prediksi
                    probability_pneumonia = float(prediction) # Pastikan tipe data float
                    
                    threshold = 0.5 # Threshold untuk klasifikasi

                    st.subheader("Hasil Prediksi:")
                    st.write(f"Probabilitas Pneumonia: {probability_pneumonia * 100:.2f}%")

                    st.subheader("Interpretasi:")
                    if probability_pneumonia >= threshold:
                        # Teks interpretasi dengan warna merah untuk Pneumonia
                        st.markdown(
                            f"<p style='color:red;'><b>Pneumonia</b></p>"
                            f"Model mengidentifikasi adanya kemungkinan pneumonia dengan probabilitas "
                            f"{probability_pneumonia * 100:.2f}%.",
                            unsafe_allow_html=True
                        )
                    else:
                        # Teks interpretasi dengan warna hijau untuk Normal
                        st.markdown(
                            f"<p style='color:green;'><b>Normal</b></p>"
                            f"Model tidak mengidentifikasi adanya tanda-tanda pneumonia (probabilitas pneumonia "
                            f"{probability_pneumonia * 100:.2f}%).",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    # Menangkap dan menampilkan error jika terjadi selama prediksi
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")