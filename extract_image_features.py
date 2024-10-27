from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

abortionImages = []
dir = 'data/images/abortion'
for img in os.listdir(dir):
    abortionImages.append(os.path.join(dir, img))
    
# https://github.com/jianjieluo/OpenAI-CLIP-Feature