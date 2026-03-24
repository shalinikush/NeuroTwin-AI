import os
import cv2
import matplotlib.pyplot as plt
import random

PROCESSED_DATA_DIR = "data/processed"


def dataset_summary():
    print("\n📊 DATASET SUMMARY")
    total_images = 0

    for tumor_class in os.listdir(PROCESSED_DATA_DIR):
        class_path = os.path.join(PROCESSED_DATA_DIR, tumor_class)

        if not os.path.isdir(class_path):
            continue

        count = len(os.listdir(class_path))
        total_images += count
        print(f"{tumor_class}: {count} images")

    print(f"\nTotal images: {total_images}")


def show_sample_images(samples_per_class=2):
    for tumor_class in os.listdir(PROCESSED_DATA_DIR):
        class_path = os.path.join(PROCESSED_DATA_DIR, tumor_class)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        if len(images) == 0:
            continue

        sample_images = random.sample(
            images, min(samples_per_class, len(images))
        )

        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            plt.imshow(img, cmap="gray")
            plt.title(f"Class: {tumor_class}")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    dataset_summary()
    show_sample_images()