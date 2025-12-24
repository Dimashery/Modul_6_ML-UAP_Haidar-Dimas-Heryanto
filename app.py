import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from PIL import Image
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Image Classification Dashboard", layout="wide")

# --- PATH CONFIGURATION ---
BASE_MODEL_PATH = r"D:\KULIAH\Semester 7\Machine Learning\PRAK\UAP\Github\model"
MODELS = {
    "CNN Custom": os.path.join(BASE_MODEL_PATH, "cnn", "cnn_model.keras"),
    "MobileNetV2": os.path.join(BASE_MODEL_PATH, "mobilenet", "mobilenetv2_model.keras"),
    "ResNet50": os.path.join(BASE_MODEL_PATH, "resnet", "resnet50_model.keras"),
    "VGG16": os.path.join(BASE_MODEL_PATH, "vgg", "vgg16_model.keras")
}

CLASS_INDICES_PATHS = {
    "CNN Custom": os.path.join(BASE_MODEL_PATH, "cnn", "cnn_model_class_indices.json"),
    "MobileNetV2": os.path.join(BASE_MODEL_PATH, "mobilenet", "mobilenetv2_model_class_indices.json"),
    "VGG16": os.path.join(BASE_MODEL_PATH, "vgg", "vgg16_model_class_indices.json"),
    "ResNet50": os.path.join(BASE_MODEL_PATH, "resnet", "resnet50_model_class_indices.json"),
}

HISTORY_PATHS = {
    "CNN Custom": os.path.join(BASE_MODEL_PATH, "cnn", "cnn_model_history.json"),
    "MobileNetV2": os.path.join(BASE_MODEL_PATH, "mobilenet", "mobilenetv2_model_history.json"),
    "VGG16": os.path.join(BASE_MODEL_PATH, "vgg", "vgg16_model_history.json"),
    "ResNet50": os.path.join(BASE_MODEL_PATH, "resnet", "resnet50_model_history.json"), # Pastikan path ini ada
}


# --- COLOR PALETTES (Gradien Minimalis) ---
MODEL_COLORS = {
    "CNN Custom": {"primary": "#6A11CB", "secondary": "#2575FC", "accent": "#BB86FC"}, # Ungu ke Biru
    "MobileNetV2": {"primary": "#00B4D8", "secondary": "#008CBA", "accent": "#90E0EF"}, # Biru Muda ke Biru Tua
    "ResNet50": {"primary": "#00C9FF", "secondary": "#92FE9D", "accent": "#6DED73"}, # Biru Langit ke Hijau Muda
    "VGG16": {"primary": "#FF7B7B", "secondary": "#F9C3A3", "accent": "#FFDAB9"}, # Merah Muda ke Orange Muda
}

# --- CUSTOM CSS UNTUK DESAIN ---
# --- CUSTOM CSS UNTUK DESAIN ---
def apply_custom_css(selected_model_name):
    colors = MODEL_COLORS.get(selected_model_name, MODEL_COLORS["CNN Custom"])
    
    st.markdown(f"""
    <style>
    /* Mengubah background utama menjadi solid dark untuk menghilangkan bar hitam */
    .stApp {{
        background-color: #0E1117;
        color: white;
    }}
    
    /* Menerapkan GRADIENT pada SIDEBAR sesuai model yang dipilih */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {colors['primary']} 0%, {colors['secondary']} 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}

    /* Menyesuaikan teks dan widget di sidebar agar tetap terbaca di atas gradien */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: white !important;
    }}
    
    /* Label Selectbox dan Radio di sidebar */
    [data-testid="stSidebar"] label {{
        color: white !important;
        font-weight: 500;
    }}

    /* Container prediksi (Metric) agar kontras dengan background solid */
    .stMetric {{
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
    }}

    /* Tombol dan Progress Bar menggunakan warna aksen model */
    .stButton>button {{
        background-color: {colors['accent']};
        color: #0E1117;
        border: none;
        font-weight: bold;
    }}
    
    .stProgress > div > div > div > div {{
        background-color: {colors['accent']} !important;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {colors['accent']} !important;
        color: {colors['accent']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_prediction_model(model_name):
    return tf.keras.models.load_model(MODELS[model_name])

def get_classes(model_name):
    path = CLASS_INDICES_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            indices = json.load(f)
            # Menangani jika format json adalah {label: index} atau {index: label}
            # Asumsi format adalah {string_label: int_index}
            return {int(v): k for k, v in indices.items()}
    # Fallback yang lebih informatif, sesuaikan dengan kelas Anda
    return {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4", 5: "Class 5", 6: "Class 6"}

def preprocess_image(image, target_size, model_name=None):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Cek model_name untuk menentukan metode preprocessing
    if model_name:
        if "MobileNetV2" in model_name:
            return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        elif "VGG16" in model_name:
            return tf.keras.applications.vgg16.preprocess_input(img_array)
        elif "ResNet50" in model_name:
            return tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # Default/Custom CNN: Rescale 1./255
    return img_array / 255.0

def load_history(model_name):
    path = HISTORY_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)
            return history
    return None

# --- UI DESIGN ---
st.sidebar.title("🚀 Model Navigator")
selected_model_name = st.sidebar.selectbox("Pilih Arsitektur Model", list(MODELS.keys()))
mode = st.sidebar.radio("Mode Pengujian", ["Single Image", "Batch Processing"])

# Terapkan CSS kustom dengan gradien sesuai model yang dipilih
apply_custom_css(selected_model_name)

st.title("🧠 AI Image Classifier")
st.markdown(f"<h3 style='color: white;'>Currently using: <span style='color:{MODEL_COLORS[selected_model_name]['accent']}'>**{selected_model_name}**</span></h3>", unsafe_allow_html=True)
st.divider()

# Load Model
model = load_prediction_model(selected_model_name)
class_labels = get_classes(selected_model_name)

# --- Deteksi otomatis ukuran input model ---
input_shape = model.input_shape
# Pastikan model memiliki input_shape yang valid, minimal 4 dimensi (batch, height, width, channels)
if len(input_shape) >= 4:
    target_size = (input_shape[1], input_shape[2])
else:
    # Fallback jika model input_shape tidak standar, misal (None, 224, 224, 3)
    # Ini sering terjadi pada model Sequential, atau jika model belum dibangun
    # Untuk amannya, kita bisa asumsikan ukuran umum 224x224 atau 128x128
    # Untuk kasus Anda, berdasarkan error sebelumnya, 128x128 adalah yang diharapkan oleh MobileNet/ResNet/VGG Anda.
    # Namun CNN Custom mungkin berbeda.
    # Jika model.input_shape tidak memberikan hasil yang diharapkan,
    # kita bisa mem-parsing dari konfigurasi model atau coba ukuran yang paling umum.
    st.warning("Could not determine exact input shape from model. Defaulting to (128, 128). "
               "Please ensure your model expects this size or adjust manually.")
    target_size = (128, 128) # Default size jika tidak dapat dideteksi

# --- TABS UNTUK PREDIKSI DAN ANALISIS ---
prediction_tab, analysis_tab = st.tabs(["Prediction", "Model Analysis"])

with prediction_tab:
    if mode == "Single Image":
        uploaded_file = st.file_uploader("Upload sebuah gambar...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            image = Image.open(uploaded_file)
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
                
            with col2:
                with st.spinner("Classifying..."):
                    processed_img = preprocess_image(image, target_size=target_size, model_name=selected_model_name)
                    preds = model.predict(processed_img)
                    result_idx = np.argmax(preds[0])
                    
                    confidence_val = float(np.max(preds[0]))
                    confidence_clipped = min(max(confidence_val, 0.0), 1.0)
                    
                    st.subheader("Result")
                    st.metric("Prediction", class_labels.get(result_idx, f"Class {result_idx}"))
                    st.write(f"Confidence: **{confidence_val * 100:.2f}%**")
                    st.progress(confidence_clipped)
                    st.caption(f"Image resized to: {target_size}")

    else: # Batch Processing
        uploaded_files = st.file_uploader("Upload beberapa gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        # --- Di dalam bagian Batch Processing ---
        if uploaded_files:
            results = []
            st.subheader("Batch Results")
            
            cols = st.columns(3)
            for idx, file in enumerate(uploaded_files):
                img = Image.open(file)
                # PERBAIKAN DI SINI: Tambahkan model_name
                processed_img = preprocess_image(img, target_size=target_size, model_name=selected_model_name)
                
                preds = model.predict(processed_img)
                res_idx = np.argmax(preds[0])
                conf = float(np.max(preds[0]))
                
                label = class_labels.get(res_idx, f"Class {res_idx}")
                results.append({"Filename": file.name, "Prediction": label, "Confidence": f"{conf*100:.2f}%"})
                
                with cols[idx % 3]:
                    st.image(img, caption=f"{label} ({conf*100:.1f}%)", use_container_width=True)
                    
            st.divider()
            st.dataframe(pd.DataFrame(results), use_container_width=True)

with analysis_tab:
    st.header("📈 Model Training History")
    history_data = load_history(selected_model_name)

    if history_data:
        epochs = range(1, len(history_data['accuracy']) + 1)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))

        # Warna aksen berdasarkan model yang dipilih
        accent_color = MODEL_COLORS[selected_model_name]['accent']
        secondary_color = MODEL_COLORS[selected_model_name]['secondary']

        # Plot Accuracy
        fig.add_trace(go.Scatter(x=list(epochs), y=history_data['accuracy'], mode='lines+markers', 
                                 name='Train Acc', marker_color=accent_color), row=1, col=1)
        if 'val_accuracy' in history_data:
            fig.add_trace(go.Scatter(x=list(epochs), y=history_data['val_accuracy'], mode='lines+markers', 
                                     name='Val Acc', marker_color='white'), row=1, col=1)

        # Plot Loss
        fig.add_trace(go.Scatter(x=list(epochs), y=history_data['loss'], mode='lines+markers', 
                                 name='Train Loss', marker_color=accent_color), row=1, col=2)
        if 'val_loss' in history_data:
            fig.add_trace(go.Scatter(x=list(epochs), y=history_data['val_loss'], mode='lines+markers', 
                                     name='Val Loss', marker_color='white'), row=1, col=2)

        # --- PERBAIKAN ERROR PAPER_BGCOLOR ---
        fig.update_layout(
            title_text=f"Training Metrics: {selected_model_name}",
            title_x=0.5,
            height=450,
            margin=dict(t=80, b=40, l=40, r=40),
            showlegend=True,
            # Gunakan RGBA untuk transparansi total agar gradien CSS terlihat
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(255,255,255,0.05)',
            font=dict(color="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Mempercantik Grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))

        st.plotly_chart(fig, use_container_width=True)
        
        # Expandable Raw Data agar tidak memenuhi layar
        with st.expander("See Raw History Data"):
            st.json(history_data)
    else:
        st.info(f"Historical data for {selected_model_name} is not available in the specified path.")