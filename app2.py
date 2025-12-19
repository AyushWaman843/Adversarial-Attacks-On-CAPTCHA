import streamlit as st
import tensorflow as tf
import numpy as np

# Load pre-trained model
@st.cache_resource
def load_model():
    try:
        model=tf.keras.models.load_model("mnist_cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Secure Login",page_icon="🔒",layout="centered")
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Secure Login</h1>", unsafe_allow_html=True)
    
    model=load_model()
    if model is None:
        st.error("Model not found. Please upload mnist_cnn_model.h5")
        return
    
    username=st.text_input("Username",key="username",help="Enter your username",max_chars=15)
    password=st.text_input("Password",type="password",key="password",help="Enter your password")
    
    if username and password:
        st.subheader("CAPTCHA Verification")
        (x_test,y_test),_=tf.keras.datasets.mnist.load_data()
        x_test=x_test.astype("float32")/255.0
        
        if "captcha_index" not in st.session_state:
            st.session_state.captcha_index=np.random.randint(0, 10000)
        
        index=st.session_state.captcha_index
        original_image=x_test[index]
        true_label=y_test[index]
        
        st.image(original_image,caption="CAPTCHA: Clean MNIST Image",width=100,channels="GRAY")
        
        if st.button("Regenerate CAPTCHA"):
            st.session_state.captcha_index=np.random.randint(0,10000)
            st.rerun()
        
        if st.button("Let Model Predict CAPTCHA"):
            pred=model.predict(np.expand_dims(original_image, 0).reshape(-1, 28, 28, 1),verbose=0)
            predicted_label=np.argmax(pred)
            st.write(f"**Model Prediction:** {predicted_label}")
            if predicted_label==true_label:
                st.success("✅ Model correctly predicted the CAPTCHA. Bot access granted.")
            else:
                st.error("❌ Model failed to predict the CAPTCHA. Automated bot access blocked.")
    
    if st.button("Login"):
        st.info("🔑 Login successful (if CAPTCHA was passed).")

if __name__=="__main__":
    main()