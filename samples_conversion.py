import os
import cv2
import numpy as np

def get_image_shapes(folder, num_images=10):
    shapes = []
    
    image_files = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]
    image_files = image_files[:num_images]  # Select first 10 images
    
    for filename in image_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            shapes.append((filename, img.shape))  # Store filename and shape
    
    return shapes

image_folder = "samples"
image_shapes = get_image_shapes(image_folder)

# Print the shapes of the first 10 images
for filename, shape in image_shapes:
    print(f"Image: {filename}, Shape: {shape}")
