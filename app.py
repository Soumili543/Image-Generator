from flask import Flask, render_template, send_file, jsonify
import os
import uuid
from PIL import Image,ImageOps
import pickle
import torch
import numpy as np


app = Flask(__name__)

# Load the .pkl file containing image tensors
model_path = "new1.pkl"
with open(model_path, "rb") as f:
    image_tensors = pickle.load(f)

# Ensure the directory for generated images exists
os.makedirs("static/generated_images", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

'''
@app.route('/generate', methods=['POST'])
def generate_image():
    # Select a random tensor from the list
    image_tensor = image_tensors[torch.randint(len(image_tensors), (1,)).item()]

    # Ensure the tensor is properly processed
    try:
        # Convert tensor to numpy array
        image_data = image_tensor.detach().numpy()

        # Normalize and scale to 0-255 range
        image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)

        # Handle different dimensional cases
        if image_data.ndim == 4:
            image_data = image_data[0]  # Remove batch dimension
        
        if image_data.ndim == 3:
            # If 3D, check if it needs channel reordering
            if image_data.shape[0] in [1, 3, 4]:  # Channel-first
                image_data = np.transpose(image_data, (1, 2, 0))
        
        # Ensure at least 3 channels for RGB
        if image_data.ndim == 2:
            image_data = np.stack([image_data]*3, axis=-1)
        
        # Create PIL Image
        image = Image.fromarray(image_data.squeeze())

        # Save the image
        image_path = f"static/generated_images/generated_image_{uuid.uuid4()}.png"
        image.save(image_path)

        return jsonify({"image_path": image_path})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500
'''
'''
working code

@app.route('/generate', methods=['POST'])
def generate_image():
    # Select a random tensor from the list
    image_tensor = image_tensors[torch.randint(len(image_tensors), (1,)).item()]

    # Process tensor into a displayable image
    try:
        image_data = image_tensor.detach().numpy()
        image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        if image_data.ndim == 4:
            image_data = image_data[0]
        if image_data.ndim == 3 and image_data.shape[0] in [1, 3, 4]:
            image_data = np.transpose(image_data, (1, 2, 0))
        if image_data.ndim == 2:
            image_data = np.stack([image_data]*3, axis=-1)
        
        image = Image.fromarray(image_data.squeeze())
        image_path = f"static/generated_images/generated_image_{uuid.uuid4()}.png"
        image.save(image_path)

        # Return image path directly for display
        return jsonify({"image_url": f"/{image_path}"})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500
'''

@app.route('/generate', methods=['POST'])
def generate_image():
    # Select a random tensor from the list
    image_tensor = image_tensors[torch.randint(len(image_tensors), (1,)).item()]

    # Process tensor into a displayable image
    try:
        image_data = image_tensor.detach().numpy()
        image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        if image_data.ndim == 4:
            image_data = image_data[0]
        if image_data.ndim == 3 and image_data.shape[0] in [1, 3, 4]:
            image_data = np.transpose(image_data, (1, 2, 0))
        if image_data.ndim == 2:
            image_data = np.stack([image_data]*3, axis=-1)
        
        image = Image.fromarray(image_data.squeeze())

        # Upscale image to 256x256
        upscale_size = (64, 64)
        image = image.resize(upscale_size, Image.NEAREST)  # NEAREST keeps it sharp

        # Save the upscaled image
        image_path = f"static/generated_images/generated_image_{uuid.uuid4()}.png"
        image.save(image_path)

        return jsonify({"image_url": f"/{image_path}"})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)