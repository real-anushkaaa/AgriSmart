
from pathlib import Path
import json
import tempfile

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = Path("data/raw/tomato_disease")
ARTIFACTS_DIR = Path("artifacts")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tf.keras.utils.set_random_seed(RANDOM_SEED)


def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    class_names = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_names) < 2:
        raise ValueError(
            "Expected at least two disease class folders."
        )

    class_to_index = {
        name: index for index, name in enumerate(class_names)
    }

    image_paths = []
    labels = []

    for class_name in class_names:
        class_dir = DATASET_DIR / class_name

        for path in sorted(class_dir.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_paths.append(str(path))
                labels.append(class_to_index[class_name])

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if len(image_paths) == 0:
        raise ValueError("No supported images found.")

    class_counts = np.bincount(labels, minlength=len(class_names))
    if class_counts.min() < 4:
        raise ValueError(
            "Each class needs at least four images for stratified train/validation/test splits."
        )

    # 70% train, 15% validation, 15% test.
    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            image_paths,
            labels,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=remaining_labels,
        )
    )

    print(f"Total images: {len(image_paths)}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")
    print(f"Number of classes: {len(class_names)}")

    print("\nLoading datasets...")
    train_ds = make_dataset(
        train_paths, train_labels, training=True
    )
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)

    # Give more weight to classes with fewer training images.
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            np.unique(train_labels), weights
        )
    }

    # MobileNetV2 pretrained on ImageNet.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    # MobileNetV2 expects pixel values scaled to [-1, 1].
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tomato-checkpoint-",
        dir=ARTIFACTS_DIR,
    ) as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "best.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        print("\nStarting training...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
        )
        best_model = tf.keras.models.load_model(checkpoint_path)

    probabilities = best_model.predict(test_ds)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=list(range(len(class_names))),
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    best_model.save(ARTIFACTS_DIR / "tomato_disease_model.keras")

    with open(
        ARTIFACTS_DIR / "tomato_disease_classes.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(class_names, file, indent=2)

    results = {
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "test_images": len(test_paths),
        "random_seed": RANDOM_SEED,
        "evaluation_note": "Single stratified holdout; interpret metrics alongside per-class support.",
    }

    with open(
        ARTIFACTS_DIR / "tomato_disease_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\nSaved model and evaluation files in artifacts/.")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
