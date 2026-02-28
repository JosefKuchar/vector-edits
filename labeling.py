import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import base64
from io import BytesIO
from time import sleep

import requests
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

from svgai.img import center_pad_image
from svgai.svg import render_fit

load_dotenv()


def encode_image_to_base64(image):
    """PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_label(image_1, image_2, model):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    image_1 = encode_image_to_base64(image_1)
    image_2 = encode_image_to_base64(image_2)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create prompt instruction for transforming the first image into the second image. Only output the prompt. Be precise and concise. Use plain language. Do not use any special characters or formatting. Do not include any additional information.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_2}"}},
            ],
        }
    ]

    payload = {"model": model, "messages": messages, "temperature": 0.0}

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response = response.json()

    return response["choices"][0]["message"]["content"]


dataset = load_from_disk("/var/tmp/xkuchar/final_dataset")
test_dataset = dataset["test"]
test_dataset = test_dataset.to_list()

for item in tqdm(test_dataset):
    image_1 = render_fit(item["item_1"]["item_svg"], 512, 512, background="white")
    image_1 = center_pad_image(image_1, 512, 512)
    image_2 = render_fit(item["item_2"]["item_svg"], 512, 512, background="white")
    image_2 = center_pad_image(image_2, 512, 512)
    image_1.save("image_1.png")
    image_2.save("image_2.png")
    model = "qwen/qwen2.5-vl-72b-instruct"

    while True:
        try:
            label = get_label(image_1, image_2, model)
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            sleep(5)
            continue
    item["instruction"] = label

# Save the test dataset with labels
test_dataset = Dataset.from_list(test_dataset)
dataset["test"] = test_dataset
#dataset.save_to_disk("/var/tmp/xkuchar/final_dataset_test_labeled")
