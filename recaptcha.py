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

# Function to load a random image from "samples"
def load_random_image(folder="samples"):
    image_files = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
    if not image_files:
        st.error("No images found in 'samples'.")
        return None, None, None
    random_file = random.choice(image_files)
    img_path = os.path.join(folder, random_file)
    
    # Load and preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 50))  # Ensure correct size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Shape: (1, 50, 200, 1)
    
    true_label = random_file.split('.')[0]  # Extract filename (label)
    
    return img, img_path, true_label

# FGSM Adversarial Attack
def fgsm_attack(model, image, epsilon=0.2):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.argmax(prediction, axis=1), prediction
        )
    
    gradient = tape.gradient(loss, image_tensor)
    sign_gradient = tf.sign(gradient)
    adversarial_image = image_tensor + epsilon * sign_gradient
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

# Function to predict an image
def predict_image(image):
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)  # Get class index
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]  # Convert back to text
    return predicted_label

# Streamlit UI - Styled Login Page
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

# Initialize session state
if 'captcha_generated' not in st.session_state:
    st.session_state['captcha_generated'] = False
if 'captcha_image' not in st.session_state:
    st.session_state['captcha_image'] = None
if 'true_label' not in st.session_state:
    st.session_state['true_label'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

# Show "Generate CAPTCHA" only if username & password are entered
if username and password:
    generate_button = st.button("Generate CAPTCHA")
    if generate_button:
        image, path, true_label = load_random_image()
        if image is not None:
            adversarial_image = fgsm_attack(model, image, epsilon=0.2)
            
            # Store in session state
            st.session_state['captcha_generated'] = True
            st.session_state['captcha_image'] = adversarial_image
            st.session_state['true_label'] = true_label
            st.session_state['prediction'] = None

# Create fixed placeholders for everything
captcha_image_placeholder = st.empty()
predict_button_placeholder = st.empty()
result_placeholder = st.empty()
regenerate_button_placeholder = st.empty()

# Always display the CAPTCHA image if generated
if st.session_state['captcha_generated']:
    captcha_image_placeholder.image(
        np.squeeze(st.session_state['captcha_image'], axis=(0, -1)),
        caption="🔒 CAPTCHA",
        use_container_width=True
    )
    
    # Prediction Button
    if predict_button_placeholder.button("Predict CAPTCHA"):
        predicted_label = predict_image(st.session_state['captcha_image'])
        st.session_state['prediction'] = predicted_label
    
    # Display prediction result if available
    if st.session_state['prediction'] is not None:
        predicted_label = st.session_state['prediction']
        true_label = st.session_state['true_label']
        
        result_text = f"Model Prediction: `{predicted_label}`\n\n"
        if str(predicted_label) == str(true_label):
            result_text += "✅ CAPTCHA Passed! Login Successful!"
        else:
            result_text += "❌ Failed CAPTCHA! Try Again."
        
        result_placeholder.markdown(result_text)
    
    # Regenerate CAPTCHA button
    if regenerate_button_placeholder.button("Generate Another Image"):
        image, path, true_label = load_random_image()
        if image is not None:
            adversarial_image = fgsm_attack(model, image, epsilon=0.2)
            
            st.session_state['captcha_image'] = adversarial_image
            st.session_state['true_label'] = true_label
            st.session_state['prediction'] = None