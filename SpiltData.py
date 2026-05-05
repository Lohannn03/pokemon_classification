import os
import shutil
import random

source_dir = "data/PokemonData"
output_dir = "data"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

random.seed(42)


for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)


for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

  
    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(output_dir, 'train', cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(output_dir, 'val', cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(output_dir, 'test', cls, img))

print("DONE")