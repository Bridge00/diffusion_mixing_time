import torch
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import torch
from diffusers import StableDiffusionPipeline


# from diffusers import AutoPipelineForText2Image


# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipeline_text2image(prompt=prompt).images[0]
# image.save('astro.png')

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
        use_auth_token=False
).to("cuda")
#
prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")


# Load CLIP
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

# Load SDXL and generate images
# Replace with your SDXL code to generate images
def generate_image_from_text(prompt):
    # Your SDXL generation code here
    image = pipe(prompt).images[0]
        
    image.save("astronaut_rides_horse.png")
    return generated_image_tensor

# Sample prompt
prompt = "a cute cat"

# Generate image from prompt using SDXL
generated_image_tensor = generate_image_from_text(prompt)

# Encode text and image with CLIP
text_inputs = clip_tokenizer(prompt, return_tensors="pt")
image_inputs = clip_processor(images=generated_image_tensor, return_tensors="pt")

# Calculate dot product similarity score
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)
    image_features = clip_model.get_image_features(**image_inputs)
    similarity_scores = (text_features @ image_features.T).squeeze()

# Get the CLIP score
clip_score = similarity_scores.item()
print("CLIP Score:", clip_score)
