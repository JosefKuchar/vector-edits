import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from skimage.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from svgai.img import center_pad_image
from svgai.svg import render_fit

output_file = "deepseek_v3"
dataset = load_from_disk(f"/var/tmp/xkuchar/editing_dataset_evals/{output_file}")

# CLIP
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

# DINOv2
processor_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=False)
model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda")

dino_similarities = []
clip_similarities = []
mse_distances = []
invalid_svg = 0
with torch.no_grad():
    for item in tqdm(dataset):
        original = item["item_2"]["item_svg"]
        edited = item["edited_svg"]

        # Render svgs to raster images
        original_rendered = render_fit(original, 512, 512, background="white")
        original_rendered = center_pad_image(original_rendered, 512, 512)
        try:
            edited_rendered = render_fit(edited, 512, 512, background="white")
            edited_rendered = center_pad_image(edited_rendered, 512, 512)
        except Exception:
            edited_rendered = Image.new("RGB", (512, 512), (255, 255, 255))
            invalid_svg += 1
            continue
        rendered_images = [original_rendered, edited_rendered]

        # CLIP
        inputs = processor_clip(images=rendered_images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
        embedding = model_clip.get_image_features(**inputs)
        embedding = embedding.cpu().numpy()

        # Calculate cosine similarity
        clip_similarities.append(cosine_similarity([embedding[0]], [embedding[1]])[0][0])

        # DINOv2
        inputs = processor_dino(images=rendered_images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
        outputs = model_dino(**inputs)
        embedding = outputs.last_hidden_state
        embedding = embedding[:, 0, :].squeeze(1)
        embedding = embedding.cpu().numpy()

        # Calculate cosine similarity
        dino_similarities.append(cosine_similarity([embedding[0]], [embedding[1]])[0][0])

        # Calculate MSE distance
        mse_distances.append(mean_squared_error(np.array(original_rendered), np.array(edited_rendered)))

print(output_file)
print("CLIP mean similarity:", np.mean(clip_similarities))
print("DINOv2 mean similarity:", np.mean(dino_similarities))
print("MSE mean distance:", np.mean(mse_distances))
print("Invalid SVGs:", invalid_svg)
