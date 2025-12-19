import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="CAPTCHA Adversarial Attack Demo", layout="wide", initial_sidebar_state="expanded")

#-------------------------------------------------------PLOTTING CAPTCHA----------------------------------------------------------------
def plot_captcha(image):
    fig, ax = plt.subplots(figsize=(8, 2))
    # Reshape if needed (if the image is 3D with a single channel)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(image.shape[0], image.shape[1])
    im = ax.imshow(image, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im)
    plt.close()
    return fig

#---------------------------------------------------------LOADING DATASET (CAPTCHA)----------------------------------------------------------------
@st.cache_data
def load_captcha_data(folder="samples", max_images=20):
    try:
        # Load label encoder
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        image_files = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
        if not image_files:
            st.error("No images found in samples folder")
            return None, None, None
        
        # Limit the number of images to process
        image_files = image_files[:max_images]
        
        images = []
        labels = []
        filenames = []
        
        for file in image_files:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 50))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)  # Shape: (50, 200, 1)
            
            true_label = file.split('.')[0]
            label_idx = label_encoder.transform([true_label])[0]
            
            images.append(img)
            labels.append(label_idx)
            filenames.append(file)
        
        return np.array(images), np.array(labels), filenames, label_encoder
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

#----------------------------------------------------FAST GRADIENT SIGN METHOD----------------------------------------------------------------
def fgsm_attack(model, image, epsilon, y_true):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image, 0))
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = SparseCategoricalCrossentropy()(tf.constant([y_true]), prediction)
    
    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = tf.clip_by_value(image_tensor + perturbation, 0, 1)
    return adversarial_image[0].numpy()

#-----------------------------------------------------PROJECTED GRADIENT METHOD----------------------------------------------------------------
def pgd_attack(model, image, epsilon, alpha, iterations, y_true):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    noise = tf.random.uniform(image_tensor.shape, -epsilon, epsilon, dtype=tf.float32)
    adversarial = tf.clip_by_value(image_tensor + noise, 0, 1)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            prediction = model(adversarial)
            loss = SparseCategoricalCrossentropy()(tf.constant([y_true], dtype=tf.int32), prediction)
        
        gradient = tape.gradient(loss, adversarial)
        signed_grad = tf.sign(gradient)
        adversarial = adversarial + alpha * signed_grad
        
        perturbation = tf.clip_by_value(adversarial - image_tensor, -epsilon, epsilon)
        adversarial = tf.clip_by_value(image_tensor + perturbation, 0, 1)
    
    return adversarial[0].numpy()

#--------------------------------------------------------IMPROVED CARLINI WAGNERS ATTACK----------------------------------------------------------------
def cw_attack(model, image, y_true, confidence=1.0, learning_rate=0.01, iterations=100, num_classes=None):
    original = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    orig_pred = model(original)
    orig_class = tf.argmax(orig_pred[0])
    
    best_adv = None
    best_dist = float('inf')
    success = False
    possible_targets = list(range(num_classes))
    if y_true in possible_targets:
        possible_targets.remove(y_true)
    
    for attempt in range(3):
        noise_scale = 0.1 * (attempt + 1)
        initial_noise = tf.random.uniform(original.shape, -noise_scale, noise_scale)
        adv_image = tf.Variable(tf.clip_by_value(original + initial_noise, 0, 1))
        
        velocity = tf.zeros_like(original)
        
        target_indices = np.random.choice(possible_targets, min(3, len(possible_targets)), replace=False)
        
        for target_class in target_indices:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            st.text(f"Attempt {attempt+1}, targeting class {target_class}")
            for i in range(iterations // 3): 
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    distortion = tf.reduce_sum(tf.square(adv_image - original))
                    
                    true_class_logit = prediction[0, y_true]
                    target_class_logit = prediction[0, target_class]
                    
                    # We want to decrease true class probability and increase target class probability
                    classification_loss = true_class_logit - target_class_logit + confidence
                    
                    # Total loss with adaptive weighting
                    # As the attack progresses, focus more on classification and less on distortion
                    c = tf.constant(10.0 * (attempt + 1), dtype=tf.float32)  # Increase weighting for later attempts
                    total_loss = distortion + c * classification_loss
                
                # Compute gradients
                gradients = tape.gradient(total_loss, adv_image)
                
                # Add momentum (helps escape local minima)
                momentum = 0.9
                velocity = momentum * velocity + (1.0 - momentum) * gradients
                
                # Apply gradients with momentum
                optimizer.apply_gradients([(velocity, adv_image)])
                
                # Ensure the adversarial example stays within valid bounds
                adv_image.assign(tf.clip_by_value(adv_image, 0, 1))
                
                # Check if current adversarial example is successful
                if i % 10 == 0 or i == iterations // 3 - 1:  # Check periodically to save computation
                    current_pred = model(adv_image)
                    current_class = tf.argmax(current_pred[0])
                    
                    # Debug progress
                    if i % 50 == 0:
                        st.text(f"Iteration {i}, current class: {current_class.numpy()}, target: {target_class}, loss: {total_loss.numpy():.4f}")
                    
                    if current_class != y_true:
                        current_dist = tf.reduce_sum(tf.square(adv_image - original)).numpy()
                        if current_dist < best_dist:
                            best_adv = adv_image.numpy()[0]
                            best_dist = current_dist
                            success = True
                            
                            # Early stopping if we found a good example
                            if best_dist < 5.0 or i > iterations // 3 - 10:
                                st.text(f"Success! Found adversarial example with distance {best_dist:.4f}")
                                break
    
    if success:
        st.text(f"Attack succeeded with distance {best_dist:.4f}")
        return best_adv

    st.text("No successful attack found, returning last attempt")
    return adv_image.numpy()[0]

@st.cache_resource
#-----------------------------------------------------------LOADING MODEL---------------------------------------------------------------
def load_model():
    try:
        model = tf.keras.models.load_model("captcha_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

#------------------------------------------------------------TRUST SCORE----------------------------------------------------------------
def calculate_confidence_score(original_image, adversarial_image):
    l2_dist = np.linalg.norm(adversarial_image - original_image)
    linf_dist = np.max(np.abs(adversarial_image - original_image))
    norm_l2 = l2_dist / np.sqrt(50 * 200)  # Normalize by image size
    norm_linf = linf_dist
    combined_score = 0.7 * np.exp(-5 * norm_l2) + 0.3 * np.exp(-10 * norm_linf)
    return combined_score

#------------------------------------------------------PLOTTING PREDICTION BARS----------------------------------------------------------------
def plot_prediction_bars(predictions, true_label, label_encoder):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(predictions[0]))
    ax.bar(x, predictions[0])
    
    # Set x-ticks to show class labels (if there aren't too many)
    if len(x) <= 20:  # Only show labels if there aren't too many
        ax.set_xticks(x)
        ax.set_xticklabels(label_encoder.classes_, rotation=45)
    else:
        # Just show a few key labels
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(label_encoder.classes_[::step], rotation=45)
    
    ax.set_ylim(0, 1)
    ax.set_xlabel('CAPTCHA Class')
    ax.set_ylabel('Probability')
    
    # Highlight true label
    ax.bar(true_label, predictions[0][true_label], color='green', alpha=0.5)
    plt.tight_layout()
    plt.close()
    return fig

#---------------------------------------------------------------TARGETED ATTACK----------------------------------------------------------------
def targeted_cw_attack(model, image, target_class, true_class, confidence=1.0, learning_rate=0.01, iterations=100):
    """
    Improved targeted Carlini-Wagner attack
    """
    # Convert to tensor
    original = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    
    # Initialize variables to track best result
    best_adv = None
    best_dist = float('inf')
    success = False
    
    # Try multiple starting points with different initializations
    for attempt in range(3):
        # Create an adversarial example with initial random noise
        noise_scale = 0.05 * (attempt + 1)  # Increase noise for each attempt
        initial_noise = tf.random.uniform(original.shape, -noise_scale, noise_scale)
        adv_image = tf.Variable(tf.clip_by_value(original + initial_noise, 0, 1))
        
        # Track velocity for momentum
        velocity = tf.zeros_like(original)
        
        # Use higher c values for targeted attacks since they're generally harder
        c_values = [5.0, 25.0, 50.0]
        
        for c in c_values:
            # Reset optimizer for each c value
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Optimization loop
            for i in range(iterations // (3 * len(c_values))):
                with tf.GradientTape() as tape:
                    tape.watch(adv_image)
                    prediction = model(adv_image)
                    
                    # Calculate distortion (L2 norm)
                    distortion = tf.reduce_sum(tf.square(adv_image - original))
                    
                    # For targeted attack: maximize target class probability and minimize others
                    target_logit = prediction[0, target_class]
                    
                    # Create mask to get all logits except target
                    mask = tf.one_hot(target_class, prediction.shape[1], on_value=0.0, off_value=1.0)
                    other_logits = prediction * tf.expand_dims(mask, 0)
                    max_other_logit = tf.reduce_max(other_logits)
                    
                    # We want target_logit > max_other_logit + confidence
                    classification_loss = tf.maximum(0.0, max_other_logit - target_logit + confidence)
                    
                    # Total loss
                    total_loss = distortion + c * classification_loss
                
                # Compute gradients
                gradients = tape.gradient(total_loss, adv_image)
                
                # Add momentum (helps escape local minima)
                momentum = 0.9
                velocity = momentum * velocity + (1.0 - momentum) * gradients
                
                # Apply gradients with momentum
                optimizer.apply_gradients([(velocity, adv_image)])
                
                # Ensure the adversarial example stays within valid bounds
                adv_image.assign(tf.clip_by_value(adv_image, 0, 1))
                
                # Check if current adversarial example is successful
                if i % 10 == 0 or i == iterations // (3 * len(c_values)) - 1:
                    current_pred = model(adv_image)
                    current_class = tf.argmax(current_pred[0])
                    
                    if current_class == target_class:
                        current_dist = tf.reduce_sum(tf.square(adv_image - original)).numpy()
                        if current_dist < best_dist:
                            best_adv = adv_image.numpy()[0]
                            best_dist = current_dist
                            success = True
                            
                            # Early stopping if we found a good example
                            if best_dist < 10.0:
                                break
    
    # If we found a successful adversarial example, return it
    if success:
        return best_adv
    
    # If no successful attack was found, return the last attempt
    return adv_image.numpy()[0]

#---------------------------------------------------------------MAIN----------------------------------------------------------------
def main():
    st.title("CAPTCHA Adversarial Attack Demo")
    
    model = load_model()
    if model is None:
        st.error("Please ensure the CAPTCHA model file (captcha_model.h5) is present in the current directory.")
        return
    
    x_test, y_test, filenames, label_encoder = load_captcha_data()
    if x_test is None or y_test is None or filenames is None:
        st.error("Failed to load CAPTCHA data.")
        return
    
    num_classes = len(label_encoder.classes_)
    
    st.sidebar.title("Attack Settings")
    attack_type = st.sidebar.selectbox(
        "Select Attack Type",
        ["FGSM", "PGD", "Carlini-Wagner"]
    )
    
    if attack_type == "FGSM":
        epsilon = st.sidebar.slider("Epsilon", 0.0, 0.25, 0.1, 0.01)
        attack_params = {"epsilon": epsilon}
    elif attack_type == "PGD":
        epsilon = st.sidebar.slider("Epsilon", 0.0, 0.25, 0.1, 0.01)
        alpha = st.sidebar.slider("Step Size", 0.001, 0.1, 0.01, 0.001)
        iterations = st.sidebar.slider("Iterations", 1, 50, 10)
        attack_params = {"epsilon": epsilon, "alpha": alpha, "iterations": iterations}
    else:  # Carlini-Wagner
        confidence = st.sidebar.slider("Confidence", 0.0, 50.0, 10.0, 1.0)
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.2, 0.05, 0.001)
        iterations = st.sidebar.slider("Iterations", 10, 1000, 200)
        attack_params = {"confidence": confidence, "learning_rate": learning_rate, "iterations": iterations}
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    
    img_index = st.number_input("Select image index (0-{})".format(len(x_test)-1), 0, len(x_test)-1, 0)
    
    cols = st.columns([1, 1, 1])
    
    with cols[0]:
        st.subheader("Original Image")
        original_image = x_test[img_index]
        with st.spinner("Generating prediction..."):
            pred = model.predict(np.expand_dims(original_image, 0), verbose=0)
        
        true_label = y_test[img_index]
        true_captcha = label_encoder.inverse_transform([true_label])[0]
        
        st.write(f"Filename: {filenames[img_index]}")
        st.write(f"True CAPTCHA: {true_captcha}")
        st.write(f"Predicted: {label_encoder.inverse_transform([np.argmax(pred[0])])[0]} (Model Confidence: {np.max(pred):.2%})")
        st.pyplot(plot_captcha(original_image))
        st.pyplot(plot_prediction_bars(pred, true_label, label_encoder))
    
    with cols[1]:
        st.subheader("Adversarial Image")
        
        # Debug info container
        if debug_mode:
            debug_container = st.container()
        
        with st.spinner("Generating adversarial example..."):
            if attack_type == "FGSM":
                adv_image = fgsm_attack(model, original_image, attack_params["epsilon"], true_label)
            elif attack_type == "PGD":
                adv_image = pgd_attack(model, original_image, attack_params["epsilon"], 
                                      attack_params["alpha"], attack_params["iterations"], true_label)
            else:
                # Only show debug output if debug mode is enabled
                if debug_mode:
                    with debug_container:
                        adv_image = cw_attack(model, original_image, true_label, 
                                            attack_params["confidence"], attack_params["learning_rate"],
                                            attack_params["iterations"], num_classes)
                else:
                    # Suppress debug output
                    adv_image = cw_attack(model, original_image, true_label, 
                                        attack_params["confidence"], attack_params["learning_rate"],
                                        attack_params["iterations"], num_classes)
            
            adv_pred = model.predict(np.expand_dims(adv_image, 0), verbose=0)
        
        confidence_score = calculate_confidence_score(original_image, adv_image)
        predicted_class = np.argmax(adv_pred[0])
        predicted_captcha = label_encoder.inverse_transform([predicted_class])[0]
        
        st.write(f"Predicted: {predicted_captcha}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Model Confidence: {np.max(adv_pred):.2%}")
        with col2:
            st.write(f"Trust Score: {confidence_score:.2%}")
        
        # Add attack results information
        st.write("### Attack Results")
        success = predicted_class != true_label
        st.write(f"Attack {'Successful' if success else 'Failed'}")
        st.write(f"Original class: {true_captcha}")
        st.write(f"New class: {predicted_captcha}")
        
        # Show top 3 predicted classes
        top_classes = np.argsort(adv_pred[0])[-3:][::-1]
        st.write("Top 3 predictions for adversarial image:")
        for i, cls in enumerate(top_classes):
            st.write(f"{i+1}. {label_encoder.inverse_transform([cls])[0]}: {adv_pred[0][cls]:.4f}")
        
        st.pyplot(plot_captcha(adv_image))
        st.pyplot(plot_prediction_bars(adv_pred, true_label, label_encoder))
    
    with cols[2]:
        st.subheader("Perturbation Analysis")
        perturbation = adv_image - original_image
        l2_dist = np.linalg.norm(perturbation)
        linf_dist = np.max(np.abs(perturbation))
        
        st.write("Perturbation Metrics:")
        st.write(f"L2 Distance: {l2_dist:.4f}")
        st.write(f"L(inf) Distance: {linf_dist:.4f}")
        
        st.pyplot(plot_captcha(perturbation))
        
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(perturbation.flatten(), bins=50)
        ax.set_title("Perturbation Distribution")
        ax.set_xlabel('Perturbation Value')
        ax.set_ylabel('Frequency')
        plt.close()
        st.pyplot(fig)
        
        # Add targeted attack option
        st.subheader("Try Targeted Attack")
        st.write("Try to target a specific class")
        available_classes = label_encoder.classes_
        target_class = st.selectbox("Select target class", available_classes)
        target_idx = label_encoder.transform([target_class])[0]
        
        if st.button("Generate Targeted Attack"):
            with st.spinner("Generating targeted adversarial example..."):
                targeted_adv = targeted_cw_attack(
                    model, original_image, target_idx, true_label,
                    confidence=attack_params["confidence"],
                    learning_rate=attack_params["learning_rate"],
                    iterations=attack_params["iterations"]
                )
                
                targeted_pred = model.predict(np.expand_dims(targeted_adv, 0), verbose=0)
                st.write(f"Target class: {target_class}")
                st.write(f"Predicted class: {label_encoder.inverse_transform([np.argmax(targeted_pred[0])])[0]}")
                st.write(f"Success: {np.argmax(targeted_pred[0]) == target_idx}")
                st.pyplot(plot_captcha(targeted_adv))
                st.pyplot(plot_prediction_bars(targeted_pred, target_idx, label_encoder))
        
        st.subheader("Model Analysis")
        if st.button("Analyze Model Sensitivity"):
            st.write("Testing model sensitivity to random noise...")
            
            num_tests = 10
            success_count = 0
            noise_levels = [0.05, 0.1, 0.2, 0.3]
            
            results = []
            
            for noise_level in noise_levels:
                level_successes = 0
                
                for _ in range(num_tests):
                    # Add random noise to the image
                    noise = np.random.uniform(-noise_level, noise_level, original_image.shape)
                    noisy_image = np.clip(original_image + noise, 0, 1)
                    
                    # Get prediction
                    noisy_pred = model.predict(np.expand_dims(noisy_image, 0), verbose=0)
                    noisy_class = np.argmax(noisy_pred[0])
                    
                    # Check if the classification changed
                    if noisy_class != true_label:
                        level_successes += 1
                
                success_rate = level_successes / num_tests
                results.append((noise_level, success_rate))
                st.write(f"Noise level {noise_level:.2f}: {success_rate:.0%} misclassification rate")
            
            # Plot results
            fig, ax = plt.subplots()
            levels, rates = zip(*results)
            ax.plot(levels, rates, marker='o')
            ax.set_xlabel('Random Noise Level')
            ax.set_ylabel('Misclassification Rate')
            ax.set_title('Model Sensitivity to Random Noise')
            ax.grid(True)
            plt.close()
            st.pyplot(fig)
            
            if max([r for _, r in results]) < 0.3:
                st.write("Your model seems quite robust to random noise, which makes adversarial attacks harder.")
                st.write("Try increasing the attack parameters or using more iterations for the Carlini-Wagner attack.")

if __name__ == "__main__":
    main()