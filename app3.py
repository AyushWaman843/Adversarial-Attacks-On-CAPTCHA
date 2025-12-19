import streamlit as st
import tensorflow as tf
import numpy as np

# Load MNIST dataset
@st.cache_data
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    return x_test, y_test

#--------------------------------------------------------------Attack--------------------------------------------------------------------
def cw_attack(model, image, y_true, confidence=5.0, learning_rate=0.1, iterations=200):
    image_tensor = tf.Variable(np.expand_dims(image, 0), dtype=tf.float32)
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    y_true_one_hot = tf.expand_dims(y_true_one_hot, 0)
    w = tf.Variable(tf.zeros_like(image_tensor))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            adversarial = 0.5 * (tf.tanh(w) + 1)
            prediction = model(adversarial)
            
            l2_loss = tf.reduce_sum(tf.square(adversarial - image_tensor))
            true_class_logit = tf.reduce_sum(y_true_one_hot * prediction, axis=1)
            other_class_logits = (1 - y_true_one_hot) * prediction - y_true_one_hot * 10000
            max_other_class_logit = tf.reduce_max(other_class_logits, axis=1)
            confidence_loss = tf.maximum(0.0, true_class_logit - max_other_class_logit + confidence)
            loss = l2_loss + confidence_loss * 10
        
        grads = tape.gradient(loss, w)
        optimizer.apply_gradients([(grads, w)])
    
    return adversarial.numpy()[0]

# Streamlit UI
def main():
    st.set_page_config(page_title="Secure Login", page_icon="🔒", layout="centered")
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Secure Login</h1>", unsafe_allow_html=True)
    
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    x_test, y_test = load_mnist()
    
    username = st.text_input("Username", key="username", help="Enter your username", max_chars=15)
    password = st.text_input("Password", type="password", key="password", help="Enter your password")
    
    if username and password:
        st.subheader("CAPTCHA Verification")
        
        if "captcha_index" not in st.session_state:
            st.session_state.captcha_index = np.random.randint(0, 10000)
        
        index=st.session_state.captcha_index
        original_image=x_test[index]
        true_label=y_test[index]
        adv_image=cw_attack(model, original_image, true_label)
        
        st.image(adv_image,caption="Adversarially Perturbed Image - Guess the Digit",width=100,channels="GRAY")
        
        user_guess=st.text_input("Enter your predicted digit (0-9):",max_chars=1)
        
        if st.button("Regenerate CAPTCHA"):
            st.session_state.captcha_index = np.random.randint(0,10000)
            st.rerun()
        
        if st.button("Submit Guess"):
            if user_guess.isdigit() and int(user_guess)==true_label:
                st.success("✅ Correct! You passed the CAPTCHA.")
                st.session_state.captcha_passed=True
            else:
                st.error("❌ Incorrect. Try again or regenerate.")
                st.session_state.captcha_passed=False
    
    if st.button("Login"):
        if "captcha_passed" in st.session_state and st.session_state.captcha_passed:
            st.success("🔑 Login successful!")
        else:
            st.error("❌ Login failed. Ensure CAPTCHA is passed.")

if __name__=="__main__":
    main()
