import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset(
    "mikronai/svg-svgrepo", split="train+valid+test", revision="3503985b648ed243f34e00587ac7af7c6da4fedd"
)
dataset = list(dataset)
print("Dataset size:", len(dataset))

from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

# CLIP
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

# DINOv2
processor_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=False)
model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda")

from itertools import batched

from svgai.img import center_pad_image
from svgai.svg import render_fit

with torch.no_grad():
    for batch in tqdm(batched(dataset, 32), total=len(dataset) // 32):
        rendered_images = []

        # Render svgs to raster images
        for item in batch:
            svg = item["item_svg"]
            rendered_image = render_fit(svg, 512, 512)
            rendered_image = center_pad_image(rendered_image, 512, 512)
            rendered_images.append(rendered_image)

        # CLIP
        inputs = processor_clip(images=rendered_images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
        embedding = model_clip.get_image_features(**inputs)
        embedding = embedding.cpu().numpy()
        for i, item in enumerate(batch):
            item["embedding_clip"] = embedding[i]

        # DINOv2
        inputs = processor_dino(images=rendered_images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
        outputs = model_dino(**inputs)
        embedding = outputs.last_hidden_state
        embedding = embedding[:, 0, :].squeeze(1)
        embedding = embedding.cpu().numpy()
        for i, item in enumerate(batch):
            item["embedding_dino"] = embedding[i]

# Create dataset and save to disk
# from datasets import Dataset

# new_dataset = Dataset.from_list(dataset)
# new_dataset.save_to_disk("/var/tmp/xkuchar/svgrepo-precomputed")
