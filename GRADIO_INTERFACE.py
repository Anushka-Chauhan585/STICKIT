interface = gr.Interface(
    fn=generate_character,  # Function to process the image
    inputs=gr.Image(type="pil"),  # Input: image (PIL format)
    outputs=gr.Image(type="pil"),  # Output: image (PIL format)
    title="Image Processing Demo",  # Optional: Title
    description="Upload an image ",  # Optional: Description
    )

# Launch the Gradio app
interface.launch()

#Launching GRADIO with our program

import torch
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
from diffusers import DiffusionPipeline
import gradio as gr

# Preprocess Image Function
def preprocess_image(image_path):
    try:
        print("Preprocessing image...")
        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Resize to 512x512 for compatibility
        image = image.resize((512, 512))

        # Convert to numpy array for OpenCV processing
        image_array = np.array(image)

        # Apply bilateral filter for smoothing
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        smoothed = cv2.bilateralFilter(image_cv2, d=15, sigmaColor=75, sigmaSpace=75)

        # Adjust brightness and contrast
        alpha = 1.2  # Contrast control
        beta = 10    # Brightness control
        enhanced = cv2.convertScaleAbs(smoothed, alpha=alpha, beta=beta)

        # Use GrabCut to isolate the foreground
        mask = np.zeros(enhanced.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (10, 10, enhanced.shape[1] - 10, enhanced.shape[0] - 10)
        cv2.grabCut(enhanced, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Modify the mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = enhanced * mask2[:, :, np.newaxis]

        # Convert back to PIL Image
        result_image_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_image_pil
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# Function for generating character
def generate_character(image_path):
    try:
        print("Processing image...")
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            print("Error processing the image")
            return "Error processing the image"

        print("Generating caption...")
        # Initialize image captioning pipeline
        captioning_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        # Generate image caption
        captions = captioning_pipe(processed_image)
        generated_caption = captions[0]['generated_text']
        print(f"Caption: {generated_caption}")

        # Create a prompt with the generated caption
        prompt = f"{generated_caption}, lonely on the beach with blue hair, 3D Disney Pixar character"
        print(f"Prompt: {prompt}")

        # Load the diffusion pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        pipeline = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix").to(device)

        print("Generating image with diffusion model...")
        # Generate the image
        result_image = pipeline(prompt=prompt, image=processed_image, strength=0.45, guidance_scale=8.5).images[0]

        return result_image

    except Exception as e:
        print(f"Error generating character: {e}")
        return "Error generating character"