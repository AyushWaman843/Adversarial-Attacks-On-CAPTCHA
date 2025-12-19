import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import random
import pickle  # For label decoding

# Load trained model
MODEL_PATH = "captcha_model.h5"
ENCODER_PATH = "label_encoder.pkl"
model = tf.keras.models.load_model(MODEL_PATH)

# Load label encoder
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
def load_random_image(folder="samples"):
    image_files = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
    if not image_files:
        st.error("No images found in 'samples'.")
        return None, None, None
    random_file = random.choice(image_files)
    img_path = os.path.join(folder, random_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 50)) 
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])
    
    true_label = random_file.split('.')[0] 
    
    return img, img_path, true_label
st.markdown(
    """
    <style>
    .login-container {
        width: 350px;
        margin: auto;
        text-align: center;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .login-container input {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .login-container button {
        width: 100%;
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="login-container">', unsafe_allow_html=True)
st.title("🔐 Secure Login")
# Login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")
st.markdown("</div>", unsafe_allow_html=True)

# Show Generate CAPTCHA only if username & password are entered
if username and password:
    if st.button("Generate CAPTCHA") or ("captcha_image" not in st.session_state):
        image, path, true_label = load_random_image()
        if image is not None:
            # Store in session
            st.session_state["captcha_image"] = image
            st.session_state["true_label"] = true_label
            st.session_state["captcha_solved"] = False  # Reset CAPTCHA status
            
            st.image(
                np.squeeze(image, axis=(0, -1)),
                caption="🔒 CAPTCHA",
                use_column_width=True,
                clamp=True
            )

# CAPTCHA input field (Only show if CAPTCHA exists)
if "captcha_image" in st.session_state:
    captcha_input = st.text_input("Enter CAPTCHA text:", key="captcha_input")
    
    if st.button("Submit CAPTCHA"):
        true_label = st.session_state["true_label"]
        
        if captcha_input.strip() == str(true_label).strip():
            st.session_state["captcha_solved"] = True
            st.success("✅ CAPTCHA Passed! Login Successful!")
        else:
            st.error("❌ Failed CAPTCHA! Try Again.")

# Regenerate CAPTCHA button (only if not solved)
if "captcha_solved" in st.session_state and not st.session_state["captcha_solved"]:
    if st.button("Generate Another Image"):
        image, path, true_label = load_random_image()
        if image is not None:
            st.session_state["captcha_image"] = image
            st.session_state["true_label"] = true_label
            
            st.image(
                np.squeeze(image, axis=(0, -1)),
                caption="🔒 CAPTCHA",
                use_column_width=True,
                clamp=True
            )