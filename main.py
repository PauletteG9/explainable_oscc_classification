
from pathlib import Path
from collections import Counter

project_folder = Path(__file__).parent
dataset_folder = project_folder / "dataset"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for split in ["train", "val", "test"]:
    print(f"\n{'=' * 35}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 35}")

    split_folder = dataset_folder / split

    for class_name in ["Normal", "OSCC"]:
        class_folder = split_folder / class_name

        if not class_folder.exists():
            print(f"{class_name}: folder not found")
            continue

        files = [
            file for file in class_folder.iterdir()
            if file.is_file()
        ]

        extensions = Counter(
            file.suffix.lower() for file in files
        )

        image_count = sum(
            count for ext, count in extensions.items()
            if ext in image_extensions
        )

        print(f"\nClass: {class_name}")
        print(f"Total files: {len(files)}")
        print(f"Recognized images: {image_count}")
        print(f"File formats: {dict(extensions)}")

