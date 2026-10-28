
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/raw/tomato_disease")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    class_counts = {}

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue

        images = [
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_counts[folder.name] = len(images)

    if not class_counts:
        print("No class folders found. Check the dataset structure.")
        return

    print("\nTOMATO DISEASE DATASET REPORT")
    print("=" * 55)

    total_images = 0

    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
        total_images += count

    print("=" * 55)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {total_images}")

    empty_classes = [
        name for name, count in class_counts.items() if count == 0
    ]

    if empty_classes:
        print("\nWARNING: These folders contain no recognized images:")
        for name in empty_classes:
            print(f"- {name}")
    else:
        print("\nAll class folders contain images.")

if __name__ == "__main__":
    main()
