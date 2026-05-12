import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# -------------------------------
# 🔇 Hide TF warnings
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# ✅ Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model('signature_cnn_final.keras')

model = load_my_model()

# -------------------------------
# ✅ Grad-CAM (FOR CUSTOM CNN)
# -------------------------------
def get_gradcam_heatmap(model, img_array, layer_name='last_conv'):

    last_conv_layer = model.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), float(predictions[0][0])

# -------------------------------
# ✅ Overlay Function
# -------------------------------
def make_overlay(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -------------------------------
# 🎯 UI
# -------------------------------
st.set_page_config(page_title="Signature Forgery Detection", layout="centered")

st.title("✍️ Signature Forgery Detection")
st.write("Upload a signature image to classify and visualize with Grad-CAM")

uploaded_file = st.file_uploader("Upload Signature Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# BEFORE UPLOAD
# -------------------------------
if uploaded_file is None:
    st.info("👆 Please upload an image to proceed")

# -------------------------------
# AFTER UPLOAD
# -------------------------------
else:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)

    st.image(img_np, caption="Original Image", use_container_width=True)

    # -------------------------------
    # 🔄 Preprocessing
    # -------------------------------
    img_resized = cv2.resize(img_np, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)

    # ⚠️ IMPORTANT: Use correct preprocessing
    # If training used rescale=1./255 → use this:
    img_array = img_array / 255.0

    # -------------------------------
    # 🔮 Prediction
    # -------------------------------
    pred = model.predict(img_array)[0][0]

    st.write("Raw prediction value:", pred)

    # ✅ Correct mapping (based on your earlier result)
    label = "Genuine ✅" if pred > 0.5 else "Forged ❌"
    confidence = pred if pred > 0.5 else 1 - pred

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # -------------------------------
    # 🔥 Grad-CAM
    # -------------------------------
    heatmap, _ = get_gradcam_heatmap(model, img_array, 'last_conv')
    overlay = make_overlay(img_np, heatmap)

    # -------------------------------
    # 📊 Display
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

    with col2:
        st.image(overlay, caption="Overlay (Model Focus)", use_container_width=True)

    st.caption("🔴 Red regions show where the model is focusing")