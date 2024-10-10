import torch
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
from diffusers import DiffusionPipeline

import numpy as np
import cv2
from PIL import Image


def preprocess_image(image_path):
    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Resize the image to 512x512 for Stable Diffusion compatibility
    image = image.resize((420, 420))

    # Convert to numpy array for OpenCV processing
    image_array = np.array(image)

    # Apply bilateral filter to smoothen the image while preserving edges
    image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    smoothed = cv2.bilateralFilter(image_cv2, d=15, sigmaColor=75, sigmaSpace=75)

    # Adjust brightness and contrast to enhance features
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(smoothed, alpha=alpha, beta=beta)

    # Create a mask initialized to probable background
    mask = np.zeros(enhanced.shape[:2], np.uint8)

    # Initialize the background and foreground models for the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle that contains the foreground object (adjust as needed)
    rect = (10, 10, enhanced.shape[1] - 10, enhanced.shape[0] - 10)  # Adjust margins as needed

    # Apply the GrabCut algorithm
    cv2.grabCut(enhanced, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask: set all definite background pixels to 0 and definite foreground pixels to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image to remove the background
    result = enhanced * mask2[:, :, np.newaxis]

    # Convert back to PIL Image
    result_image_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_image_pil

# Example usage
image_path = "shanaaya.jpg"  # Replace with the path to your image
processed_image = preprocess_image(image_path)


# Initialize image captioning pipeline
captioning_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Generate image caption
captions = captioning_pipe(processed_image)
generated_caption = captions[0]['generated_text']


# Create a prompt combining the generated caption and style descriptor
prompt = f"{generated_caption}, retain facial features, Disney Pixar character"

print(prompt)

# Load the diffusion pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix").to(device)

# Generate a Pixar-Disney style character based on the prompt
result_image = pipeline(prompt=prompt, image=processed_image, strength=0.45, guidance_scale=8.5).images[0]

# Save and Display the Result
result_image.save("pixar_disney_character_output.jpg")
result_image.show()

#Tried inculcating OPENAI for User Prompts (under developed)

import openai

openai.api_key = '#given API Key'

response = openai.Completion.create(
    model="code-davinci-002",
    prompt="describe a scenery of good night",
    max_tokens=100
)

print(response.choices[0].text.strip())