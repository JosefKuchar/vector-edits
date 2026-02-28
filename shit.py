import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from datasets import load_from_disk

from svgai.img import center_pad_image
from svgai.svg import render_fit

output_file = "gemini_flash"
dataset = load_from_disk(f"/var/tmp/xkuchar/editing_dataset_evals/{output_file}")

# 1900
# 1601


# 1901


item = dataset[1901]
original = item["item_1"]["item_svg"]
edited = item["item_2"]["item_svg"]
print(item["instruction"])
print(original)
print(edited)
print(item["edited_svg"])

original_rendered = render_fit(original, 512, 512, background="white")
original_rendered = center_pad_image(original_rendered, 512, 512)
edited_rendered = render_fit(edited, 512, 512, background="white")
edited_rendered = center_pad_image(edited_rendered, 512, 512)

original_rendered.save("1.png")
edited_rendered.save("2.png")
