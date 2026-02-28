import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
from time import sleep

import requests
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

dataset = load_from_disk("/var/tmp/xkuchar/final_dataset_test_labeled")
test_dataset = dataset["test"]
test_dataset = test_dataset.to_list()


model = "deepseek/deepseek-chat-v3-0324"
output_file = "deepseek_v3"


def get_edited_svg(svg, instruction):
    prompt = f"Change the provided svg according to this instruction: {instruction}\nOutput the full modified svg. Only output the svg, nothing else.\n\n{svg}"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    payload = {"model": model, "messages": messages, "temperature": 0.0}

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response = response.json()

    return response["choices"][0]["message"]["content"]


for item in tqdm(test_dataset):
    svg = item["item_1"]["item_svg"]
    instruction = item["instruction"]

    while True:
        try:
            edited_svg = get_edited_svg(svg, instruction)
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            sleep(5)

    # remove everything before the first <svg> tag
    edited_svg = re.sub(r".*<svg", "<svg", edited_svg, flags=re.DOTALL)
    # remove everything after the last </svg> tag
    edited_svg = re.sub(r"</svg>.*", "</svg>", edited_svg, flags=re.DOTALL)

    item["edited_svg"] = edited_svg

dataset_with_edited_svg = Dataset.from_list(test_dataset)
dataset_with_edited_svg.save_to_disk(f"/var/tmp/xkuchar/editing_dataset_evals/{output_file}")
