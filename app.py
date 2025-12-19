import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # type: ignore
import time
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
st.set_page_config(page_title="MNIST Adversarial Attack Demo",layout="wide",initial_sidebar_state="expanded") #our heading for Streamlit app

#-------------------------------------------------------GENERATING DIGIT----------------------------------------------------------------
def plot_digit(image): #plotting a digit from the mnist dataset
    fig,ax=plt.subplots(figsize=(4, 4))
    im=ax.imshow(image, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im)
    plt.close()
    return fig

# Loading and preprocessing MNIST data
@st.cache_data
#---------------------------------------------------------LOADING DATASET (MNIST)----------------------------------------------------------------
def load_mnist_data():
    (x_train,y_train),(x_test, y_test)=mnist.load_data()
    x_test=x_test.astype('float32')/255.0
    return x_test,y_test

#----------------------------------------------------FAST GRADIENT SIGN METHOD----------------------------------------------------------------
def fgsm_attack(model,image,epsilon,y_true): #FGSM ATTACK
    image_tensor=tf.convert_to_tensor(np.expand_dims(image,0))
    y_true_one_hot=tf.one_hot(y_true,depth=10)
    y_true_one_hot=tf.expand_dims(y_true_one_hot,0)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction=model(image_tensor)
        target_class=tf.argmax((1-y_true_one_hot)*prediction,axis=1)
        target_one_hot=tf.one_hot(target_class,depth=10)
        loss=-CategoricalCrossentropy()(target_one_hot,prediction)  # Negative because we want to maximize this loss
    gradient=tape.gradient(loss,image_tensor)
    perturbation=epsilon*tf.sign(gradient)
    adversarial_image=tf.clip_by_value(image_tensor + perturbation, 0, 1)
    return adversarial_image[0].numpy()

#-----------------------------------------------------PROJECTED GRADIENT METHOD----------------------------------------------------------------
def pgd_attack(model,image,epsilon,alpha,iterations,y_true):
    image_tensor=tf.convert_to_tensor(np.expand_dims(image,0))
    y_true_one_hot=tf.one_hot(y_true,depth=10)
    y_true_one_hot=tf.expand_dims(y_true_one_hot,0)
    
    noise=tf.random.uniform(image_tensor.shape,-epsilon,epsilon)
    adversarial=tf.clip_by_value(image_tensor+noise,0,1)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            prediction=model(adversarial)
            target_class=tf.argmax((1-y_true_one_hot)*prediction,axis=1)
            target_one_hot=tf.one_hot(target_class, depth=10)
            loss=-CategoricalCrossentropy()(target_one_hot, prediction)
        
        gradient=tape.gradient(loss,adversarial)
        signed_grad=tf.sign(gradient)
        adversarial=adversarial+alpha*signed_grad
        
        perturbation=tf.clip_by_value(adversarial-image_tensor,-epsilon,epsilon)
        adversarial=tf.clip_by_value(image_tensor+perturbation,0,1)
    
    return adversarial[0].numpy()

#--------------------------------------------------------CARLINI WAGNERS ATTACK----------------------------------------------------------------
def cw_attack(model, image, y_true, confidence=0.0, learning_rate=0.01, iterations=100):
    image_tensor = tf.Variable(np.expand_dims(image, 0), dtype=tf.float32)
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    y_true_one_hot = tf.expand_dims(y_true_one_hot, 0)
    w = tf.Variable(tf.zeros_like(image_tensor))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    best_adv = None
    best_loss = float('inf')
    
    for i in range(iterations):
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
        
        current_pred = prediction.numpy()
        current_loss = loss.numpy()

        if current_loss < best_loss:
            if np.argmax(current_pred[0]) != y_true:
                best_loss = current_loss
                best_adv = adversarial.numpy()[0]
    if best_adv is not None:
        return best_adv
    else:\
        return adversarial.numpy()[0]

@st.cache_resource
#-----------------------------------------------------------LOADING MODEL---------------------------------------------------------------
def load_model():
    try:
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

#------------------------------------------------------------TRUST SCORE----------------------------------------------------------------
def calculate_confidence_score(original_image, adversarial_image):
    l2_dist = np.linalg.norm(adversarial_image - original_image)
    linf_dist = np.max(np.abs(adversarial_image - original_image))
    norm_l2 = l2_dist / np.sqrt(784)
    norm_linf = linf_dist
    combined_score=0.7*np.exp(-5*norm_l2)+0.3*np.exp(-10*norm_linf)
    return combined_score

#------------------------------------------------------PLOTTING PREDICTION BARS----------------------------------------------------------------
def plot_prediction_bars(predictions, true_label):
    fig,ax=plt.subplots(figsize=(4, 2))
    x=np.arange(10)
    ax.bar(x,predictions[0])
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    #Highlight true label
    ax.bar(true_label,predictions[0][true_label],color='green',alpha=0.5)
    plt.close()
    return fig

#---------------------------------------------------------------MAIN----------------------------------------------------------------
def main():
    st.title("MNIST Adversarial Attack Demo")
    model=load_model()
    if model is None:
        st.error("Please ensure the MNIST model fileQ (mnist_cnn_model.h5) is present in the current directory.")
        return
    x_test,y_test=load_mnist_data()
    st.sidebar.title("Attack Settings")
    attack_type=st.sidebar.selectbox(
        "Select Attack Type",
        ["FGSM","PGD","Carlini-Wagner"]
    )
    if attack_type=="FGSM":
        epsilon=st.sidebar.slider("Epsilon", 0.0, 0.25, 0.1, 0.01)
        attack_params={"epsilon": epsilon}
    elif attack_type=="PGD":
        epsilon=st.sidebar.slider("Epsilon", 0.0, 0.25, 0.1, 0.01)
        alpha=st.sidebar.slider("Step Size", 0.001, 0.1, 0.01, 0.001)
        iterations=st.sidebar.slider("Iterations", 1,50,10)
        attack_params={"epsilon": epsilon,"alpha": alpha,"iterations": iterations}
    else:  #Carlini-wagner
        confidence=st.sidebar.slider("Confidence",0.0,5.0,1.0,0.1)
        learning_rate=st.sidebar.slider("Learning Rate",0.001,0.1,0.01,0.001)
        iterations=st.sidebar.slider("Iterations",10,200,100)
        attack_params={"confidence": confidence,"learning_rate":learning_rate,"iterations":iterations}
    img_index=st.number_input("Select image index (0-9999)",0,9999,0)
    cols=st.columns([1, 1, 1])
    with cols[0]:
        st.subheader("Original Image")
        original_image=x_test[img_index]
        with st.spinner("Generating prediction..."):
            pred = model.predict(np.expand_dims(original_image, 0), verbose=0)
        st.write(f"True Label: {y_test[img_index]}")
        st.write(f"Predicted: {np.argmax(pred)} (Model Confidence: {np.max(pred):.2%})")
        st.pyplot(plot_digit(original_image))
        st.pyplot(plot_prediction_bars(pred, y_test[img_index]))
    with cols[1]:
        st.subheader("Adversarial Image")
        with st.spinner("Generating adversarial example..."):
            if attack_type == "FGSM":
                adv_image = fgsm_attack(model, original_image, attack_params["epsilon"], y_test[img_index])
            elif attack_type == "PGD":
                adv_image = pgd_attack(model, original_image, attack_params["epsilon"], attack_params["alpha"], attack_params["iterations"], y_test[img_index])
            else:
                adv_image=cw_attack(model, original_image, y_test[img_index],attack_params["confidence"], attack_params["learning_rate"],attack_params["iterations"])
            
            adv_pred=model.predict(np.expand_dims(adv_image, 0), verbose=0)
        confidence_score=calculate_confidence_score(original_image, adv_image)
        st.write(f"Predicted: {np.argmax(adv_pred)}")
        col1,col2=st.columns(2)
        with col1:
            st.write(f"Model Confidence: {np.max(adv_pred):.2%}")
        with col2:
            st.write(f"Trust Score: {confidence_score:.2%}")
        st.pyplot(plot_digit(adv_image))
        st.pyplot(plot_prediction_bars(adv_pred, y_test[img_index]))
    with cols[2]:
        st.subheader("Perturbation Analysis")
        perturbation=adv_image-original_image
        l2_dist=np.linalg.norm(perturbation)
        linf_dist=np.max(np.abs(perturbation))
        st.write("Perturbation Metrics:")
        st.write(f"L2 Distance: {l2_dist:.4f}")
        st.write(f"L(inf) Distance: {linf_dist:.4f}")
        st.pyplot(plot_digit(perturbation))
        fig,ax=plt.subplots(figsize=(4,2))
        ax.hist(perturbation.flatten(),bins=50)
        ax.set_title("Perturbation Distribution")
        ax.set_xlabel('Perturbation Value')
        ax.set_ylabel('Frequency')
        plt.close()
        st.pyplot(fig)
if __name__=="__main__":
    main()