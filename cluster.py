import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from svgai.img import center_pad_image
from svgai.svg import render_fit

load_dotenv()

from datasets import load_dataset

dataset = load_dataset(
    "mikronai/svg-svgrepo", split="train+valid+test", revision="3503985b648ed243f34e00587ac7af7c6da4fedd"
)
dataset = dataset.to_list()
print("Dataset size:", len(dataset))

collections = {}
for item in tqdm(dataset):
    slug = item["collection_slug"]
    if slug not in collections:
        collections[slug] = []
    collections[slug].append(item)
print(f"Number of collections: {len(collections)}")

from transformers import CLIPModel, CLIPProcessor

processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
model_clip.eval()
pass


from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

treshold = 0.94
treshold_max = 0.9995  # these images are identical
with torch.no_grad():
    for slug in tqdm(collections):
        collection = collections[slug]
        embeds = []
        dataset = []
        for item in collection:
            svg = item["item_svg"]
            img = render_fit(svg, 512, 512, background="white")
            img = center_pad_image(img, 512, 512)
            inputs = processor_clip(images=img, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            embedding = model_clip.get_image_features(**inputs)
            embedding = embedding.cpu().numpy().flatten()
            embeds.append(embedding)
        similarities = cosine_similarity(embeds)
        for i in range(similarities.shape[0]):
            for j in range(similarities.shape[1]):
                if i == j:
                    continue
                if similarities[i][j] >= treshold and similarities[i][j] < treshold_max:
                    dataset.append(
                        {
                            "collection_slug": slug,
                            "item_1": collection[i],
                            "item_2": collection[j],
                            "similarity": similarities[i][j],
                        }
                    )
        if len(dataset) == 0:
            print(f"Collection {slug} has no similar items")
            continue
        dataset = Dataset.from_list(dataset)
        dataset.save_to_disk(f"/var/tmp/xkuchar/editing_dataset/{slug}")
