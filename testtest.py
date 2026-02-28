from datasets import load_dataset

dataset = load_dataset("mikronai/svg-svgrepo")
print(dataset)
print("train", len(dataset["train"]))
print("validation", len(dataset["valid"]))
print("test", len(dataset["test"]))
print(dataset["train"][0])
