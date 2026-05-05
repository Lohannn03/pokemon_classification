from torchvision import datasets

dataset = datasets.ImageFolder("data/train")
with open("classes.txt", "w") as f:
    for cls in dataset.classes:
        f.write(cls + "\n")