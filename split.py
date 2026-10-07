import os
import random
import shutil
from pathlib import Path

random.seed(42)

source_dir = "F:\R&D_Work\Final_Year_PCOS\PCOS"
output_dir = "F:\R&D_Work\Final_Year_PCOS\PCOS_Split"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

classes = ["infected", "noninfected"]

for cls in classes:

    source_class = os.path.join(source_dir, cls)

    images = [
        f for f in os.listdir(source_class)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = images[:train_count]
    val_files = images[train_count:train_count + val_count]
    test_files = images[train_count + val_count:]

    for split, files in {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }.items():

        split_dir = os.path.join(output_dir, split, cls)
        Path(split_dir).mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy(
                os.path.join(source_class, file),
                os.path.join(split_dir, file)
            )

    print(f"\n{cls}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")