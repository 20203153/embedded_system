import cv2
import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
import pandas as pd

# Global variable to store click coordinates
click_coordinates = []

# Mouse click event handler
def save_click_event(event, x, y, flags, param):
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates = [(x, y)]
        print(f"Clicked at: ({x}, {y})")

# Coordinate transformation function
def transform_coordinates(coordinates, transformation, original_size, augmented_size):
    x, y = coordinates
    original_height, original_width = original_size
    aug_height, aug_width = augmented_size

    if transformation == "flipped_lr":
        x = original_width - x - 1
    elif transformation == "flipped_ud":
        y = original_height - y - 1
    elif transformation == "flipped_both":
        x = original_width - x - 1
        y = original_height - y - 1
    elif transformation == "rotated_90":
        x, y = y, original_width - x - 1
    elif transformation == "rotated_180":
        x = original_width - x - 1
        y = original_height - y - 1
    elif transformation == "rotated_270":
        x, y = original_height - y - 1, x

    return int(min(max(x, 0), aug_width - 1)), int(min(max(y, 0), aug_height - 1))

# Image augmentation function
def augment_image(image):
    augmented_images = []
    transformations = []

    # Original image
    augmented_images.append(image)
    transformations.append("original")

    # Flip left-right
    flipped_lr = cv2.flip(image, 1)
    augmented_images.append(flipped_lr)
    transformations.append("flipped_lr")

    # Flip up-down
    flipped_ud = cv2.flip(image, 0)
    augmented_images.append(flipped_ud)
    transformations.append("flipped_ud")

    # Flip both
    flipped_both = cv2.flip(image, -1)
    augmented_images.append(flipped_both)
    transformations.append("flipped_both")

    # Rotate 90 degrees
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated_90)
    transformations.append("rotated_90")

    # Rotate 180 degrees
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_images.append(rotated_180)
    transformations.append("rotated_180")

    # Rotate 270 degrees
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(rotated_270)
    transformations.append("rotated_270")

    return augmented_images, transformations

# Function to generate augmented filenames
def get_augmented_filename(filename, transformation):
    name, ext = os.path.splitext(filename)
    return f"{name}_{transformation}{ext}"

# Function for labeling images
def label_images():
    # Select image folder
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Image Folder")
    if not folder_selected:
        print("No folder selected.")
        return

    # Ask where to save labels
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Labels As")
    if not save_path:
        print("No save path selected.")
        return

    labeled_data = []

    image_files = [f for f in os.listdir(folder_selected) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the selected folder.")
        return

    for image_file in image_files:
        img_path = os.path.join(folder_selected, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", save_click_event)
        global click_coordinates
        click_coordinates = []

        print(f"Click on the image to select a point for {image_file}. Press 'N' to skip, 'ESC' to exit.")

        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF
            if click_coordinates:
                x, y = click_coordinates[0]
                print(f"Selected point: ({x}, {y}) on image {image_file}")
                labeled_data.append([image_file, x, y])
                click_coordinates = []
                break
            elif key == ord('n'):
                print(f"Skipping image {image_file}")
                break
            elif key == 27:  # ESC key
                print("Exiting labeling.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    if labeled_data:
        df = pd.DataFrame(labeled_data, columns=['image', 'x', 'y'])
        df.to_csv(save_path, index=False)
        print(f"Labels saved to {save_path}")
    else:
        print("No labels to save.")

# Function for augmenting images
def augment_images():
    # Select CSV file with labels
    root = tk.Tk()
    root.withdraw()
    csv_file = filedialog.askopenfilename(title="Select CSV file with labels", filetypes=[("CSV files", "*.csv")])
    if not csv_file:
        print("No CSV file selected.")
        return

    # Read labels
    labels_df = pd.read_csv(csv_file)

    # Select image folder
    folder_selected = filedialog.askdirectory(title="Select Image Folder")
    if not folder_selected:
        print("No folder selected.")
        return

    # Ask where to save augmented images and labels
    save_folder = filedialog.askdirectory(title="Select folder to save augmented images")
    if not save_folder:
        print("No save folder selected.")
        return

    augmented_labels = []

    for idx, row in labels_df.iterrows():
        image_file = row['image']
        x = row['x']
        y = row['y']

        img_path = os.path.join(folder_selected, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        augmented_images, transformations = augment_image(img)

        for aug_img, transformation in zip(augmented_images, transformations):
            aug_filename = get_augmented_filename(image_file, transformation)
            aug_img_path = os.path.join(save_folder, aug_filename)

            aug_height, aug_width = aug_img.shape[:2]
            new_x, new_y = transform_coordinates((x, y), transformation, (img_height, img_width), (aug_height, aug_width))

            augmented_labels.append([aug_filename, new_x, new_y])

            cv2.imwrite(aug_img_path, aug_img)

    if augmented_labels:
        aug_labels_df = pd.DataFrame(augmented_labels, columns=['image', 'x', 'y'])
        # Save augmented labels
        save_labels_path = os.path.join(save_folder, "augmented_labels.csv")
        aug_labels_df.to_csv(save_labels_path, index=False)
        print(f"Augmented images and labels saved to {save_folder}")
    else:
        print("No augmented images generated.")

if __name__ == "__main__":
    mode = input("Choose mode (label/augment): ").strip().lower()
    if mode == "label":
        label_images()
    elif mode == "augment":
        augment_images()
    else:
        print("Invalid mode selected. Please use 'label' or 'augment'.")
