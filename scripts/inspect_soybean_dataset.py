from pathlib import Path

DATASET_DIR = Path("data/raw/soybean_disease")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not DATASET_DIR.is_dir():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    class_counts = {}

    for folder in sorted(DATASET_DIR.iterdir()):
        if folder.is_dir():
            count = sum(
                1
                for path in folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            class_counts[folder.name] = count

    if not class_counts:
        print("No class folders found.")
        return

    print("\nSOYBEAN DISEASE DATASET REPORT")
    print("=" * 50)

    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    print("=" * 50)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")

    empty_classes = [
        name for name, count in class_counts.items() if count == 0
    ]

    if empty_classes:
        print("\nWARNING: Empty or unrecognized image folders:")
        for name in empty_classes:
            print(f"- {name}")
    else:
        print("\nAll class folders contain images.")


if __name__ == "__main__":
    main()
