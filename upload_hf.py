from datasets import load_from_disk

dataset = load_from_disk("/var/tmp/xkuchar/final_dataset_test_labeled")

# add instruction column to validation and train
new_column = [""] * len(dataset["train"])
dataset["train"] = dataset["train"].add_column("instruction", new_column)
new_column = [""] * len(dataset["validation"])
dataset["validation"] = dataset["validation"].add_column("instruction", new_column)

dataset.push_to_hub("authoranonymous321/EditSVGDataset", token="")
