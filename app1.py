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

# Carlini-Wagner Attack
@st.cache_data
#------------------------------------------------------------ATTACK-------------------------------------------------------------------------
def cw_attack(model, image, y_true, confidence=5.0, learning_rate=0.1, iterations=200):
    image_tensor = tf.Variable(np.expand_dims(image, 0), dtype=tf.float32)
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    y_true_one_hot = tf.expand_dims(y_true_one_hot, 0)
    w = tf.Variable(tf.zeros_like(image_tensor))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    best_adv = None
    best_loss = float('inf')
    
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
        if loss.numpy() < best_loss and np.argmax(prediction.numpy()) != y_true:
            best_loss = loss.numpy()
            best_adv = adversarial.numpy()[0]
    return best_adv if best_adv is not None else adversarial.numpy()[0]

#------------------------------------------------------------Streamlit UI-------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Secure Login",layout="centered")
    st.markdown("<h1 style='text-align: center;color: #4A90E2;'>Secure Login</h1>", unsafe_allow_html=True)
    
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
            st.session_state.captcha_index = np.random.randint(0, 10000)
        
        index = st.session_state.captcha_index
        original_image = x_test[index]
        true_label = y_test[index]
        adv_image = cw_attack(model, original_image, true_label)
        
        st.image(adv_image, caption="CAPTCHA: Adversarially Perturbed Image", width=100, channels="GRAY")
        
        if st.button("Regenerate CAPTCHA"):
            st.session_state.captcha_index = np.random.randint(0, 10000)
            st.rerun()
        
        if st.button("Let Model Predict CAPTCHA"):
            pred=model.predict(np.expand_dims(adv_image,0),verbose=0)
            predicted_label=np.argmax(pred)
            st.write(f"**Model Prediction:**{predicted_label}")
            if predicted_label==true_label:
                st.success("✅ Model correctly predicted the CAPTCHA. Bot access granted.")
            else:
                st.error("❌ Model failed to predict the CAPTCHA. Automated bot access blocked.")
    
    if st.button("Login"):
        st.info("🔑 Login successful (if CAPTCHA was passed).")

if __name__=="__main__":
    main()
