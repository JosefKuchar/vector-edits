import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_from_disk
from dotenv import load_dotenv

from svgai.img import center_pad_image
from svgai.svg import render_fit

load_dotenv()

dataset = load_from_disk("/var/tmp/xkuchar/final_dataset_test_labeled")
test_dataset = dataset["test"]


item = test_dataset[170]
print(item["instruction"])
image_1 = render_fit(item["item_1"]["item_svg"], 512, 512, background="white")
image_1 = center_pad_image(image_1, 512, 512)
image_2 = render_fit(item["item_2"]["item_svg"], 512, 512, background="white")
image_2 = center_pad_image(image_2, 512, 512)
print(item["item_1"]["item_svg"])
print(item["item_2"]["item_svg"])
image_1.save("image_1.png")
image_2.save("image_2.png")
