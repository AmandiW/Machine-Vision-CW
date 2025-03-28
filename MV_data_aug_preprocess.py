import os
import shutil
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
excel_path = r"D:\APTOS data\fyp_image_labels.csv"
image_folder = r"D:\APTOS data\fyp_images"
processed_folder = r"D:\APTOS data\fyp_processed_dataset"
output_images_folder = os.path.join(processed_folder, "images")

# Create directories for the processed dataset
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(output_images_folder, exist_ok=True)

# Load the dataset
data = pd.read_csv(excel_path)

# Inspect the data
print("Initial Dataset:")
print(data.head())
print(f"Total rows: {data.shape[0]}")

# Check for missing values
print(f"Dataset before removing missing values: {data.shape[0]} rows")
missing_values = data.isnull().sum()
print(missing_values)

# Remove rows with missing id_code or diagnosis
data = data.dropna(subset=['id_code', 'diagnosis'])
print(f"Dataset after removing missing values: {data.shape[0]} rows")

# Check for duplicate rows
print("\nChecking for duplicates...")
duplicates = data.duplicated().sum()
print(f"Duplicates found: {duplicates}")

# Remove duplicates
data = data.drop_duplicates()
print(f"Dataset after removing duplicates: {data.shape[0]} rows")

# Check for missing images
print("\nChecking for missing or corrupted images...")
missing_images = []
for img_id in data['id_code']:
    img_path = os.path.join(image_folder, f"{img_id}.png")
    if not os.path.exists(img_path):
        missing_images.append(img_id)

print(f"Missing images: {len(missing_images)}")

# Remove rows corresponding to missing images
data = data[~data['id_code'].isin(missing_images)]
print(f"Dataset after removing missing images: {data.shape[0]} rows")

# Saving the cleaned dataset to a CSV file
cleaned_csv_path = os.path.join(processed_folder, "cleaned_image_labels.csv")
data.to_csv(cleaned_csv_path, index=False)
print(f"\nCleaned dataset saved to: {cleaned_csv_path}")
print(f"Processed images saved to: {output_images_folder}")

# Load the cleaned dataset to perform augmentation
data = pd.read_csv(cleaned_csv_path)

# Class distribution
class_counts = data['diagnosis'].value_counts()
print("Class Distribution:")
print(class_counts)

# Visualize the distribution
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Class Distribution")
plt.xlabel("Diagnosis (0: No DR, 1: DR)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()


# Paths to save augmented images
original_csv_path = r"D:\APTOS data\fyp_processed_dataset\cleaned_image_labels.csv"
image_folder = r"D:\APTOS data\fyp_processed_dataset\images"
augmented_folder = r"D:\APTOS data\fyp_processed_dataset\augmented_images"
os.makedirs(augmented_folder, exist_ok=True)

# Load the original dataset
data = pd.read_csv(original_csv_path)

def pad_and_resize(img, target_size=(224, 224)):
    h, w, _ = img.shape
    scale = target_size[0] / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the image while maintaining aspect ratio
    resized = cv2.resize(img, (new_w, new_h))

    # Add padding to reach target size
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotate images randomly up to 30 degrees
    zoom_range=0.2,             # Slight zoom in/out
    horizontal_flip=True,       # Flip images horizontally
    vertical_flip=False,        # No vertical flip for medical images
    fill_mode='nearest'         # Fill empty areas with nearest pixel values
)

# Number of augmented images per original image
augment_count = 2

print("Generating augmented images...")
for img_name in tqdm(os.listdir(image_folder), desc="Augmenting images"):
    img_path = os.path.join(image_folder, img_name)
    if img_path.endswith(".png"):
        img = cv2.imread(img_path)  # Read the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = pad_and_resize(img)  # Pad and resize to maintain aspect ratio
        img = img.reshape((1,) + img.shape)  # Reshape for the generator

        # Save augmented images
        aug_iter = datagen.flow(img, batch_size=1, save_to_dir=augmented_folder,
                                save_prefix=f"{os.path.splitext(img_name)[0]}_aug", save_format='png')
        for _ in range(augment_count):
            next(aug_iter)

# Create new entries for augmented images
augmented_data = []
for img_name in os.listdir(augmented_folder):
    if img_name.endswith(".png"):
        # Extract the original ID from the augmented filename
        original_id = img_name.split("_aug")[0]
        # Find the diagnosis for the original image
        label = data[data['id_code'] == original_id]['diagnosis'].values[0]
        # Append a new entry for the augmented image
        augmented_data.append({"id_code": os.path.splitext(img_name)[0], "diagnosis": label})

# Convert to DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Combine with the original dataset
updated_data = pd.concat([data, augmented_df], ignore_index=True)

# Save the updated dataset
updated_csv_path = r"D:\APTOS data\fyp_processed_dataset\cleaned_image_labels.csv"
updated_data.to_csv(updated_csv_path, index=False)

print(f"Updated dataset saved to: {updated_csv_path}")
print(f"Total images in updated dataset: {updated_data.shape[0]}")

# Verify the dimensions of augmented images
incorrect_dimensions = []
for img_name in os.listdir(augmented_folder):
    img_path = os.path.join(augmented_folder, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    if (h, w) != (224, 224):
        incorrect_dimensions.append((img_name, h, w))

if len(incorrect_dimensions) == 0:
    print("All images have correct dimensions (224x224).")
else:
    print(f"{len(incorrect_dimensions)} images have incorrect dimensions:")
    for name, h, w in incorrect_dimensions[:10]:  # Print the first 10
        print(f"{name}: {h}x{w}")

# check if the augmented dataset is valid
# Paths
csv_path = r"D:\APTOS data\fyp_processed_dataset\cleaned_image_labels.csv"
original_images_folder = r"D:\APTOS data\fyp_processed_dataset\images"
augmented_images_folder = r"D:\MV_CW_Data\combined_images"

# Load the dataset
data = pd.read_csv(csv_path)

# Inspect the dataset
print("Dataset loaded:")
print(data.head())

# Paths
csv_path = r"D:\MV_CW_Data\cleaned_valid_image_labels.csv"

# Load the CSV file
data = pd.read_csv(csv_path)

# Get IDs from the CSV
csv_ids = set(data['id_code'])

# Get IDs from the original images folder
original_ids = set(os.path.splitext(f)[0] for f in os.listdir(original_images_folder) if f.endswith(".png"))

# Get IDs from the augmented images folder
augmented_ids = set(os.path.splitext(f)[0] for f in os.listdir(augmented_images_folder) if f.endswith(".png"))

# Combine folder IDs
all_folder_ids = original_ids.union(augmented_ids)

# Check for missing IDs in folders
missing_in_folders = csv_ids - all_folder_ids

# Check for extra IDs in folders
extra_in_folders = all_folder_ids - csv_ids

# Results
if not missing_in_folders:
    print("All images listed in the CSV are present in the folders.")
else:
    print(f"Missing images in folders: {len(missing_in_folders)}")
    print(f"Missing IDs (first 10): {list(missing_in_folders)[:10]}")
    # Save missing IDs to a text file
    with open(r"D:\APTOS data\missing_ids_in_folders.txt", "w") as file:
        file.write("\n".join(missing_in_folders))

if not extra_in_folders:
    print("No extra images are present in the folders.")
else:
    print(f"Extra images in folders: {len(extra_in_folders)}")
    print(f"Extra IDs (first 10): {list(extra_in_folders)[:10]}")
    # Save extra IDs to a text file
    with open(r"D:\APTOS data\extra_ids_in_folders.txt", "w") as file:
        file.write("\n".join(extra_in_folders))

# Final Dataset Check
print(f"Number of images in CSV: {len(csv_ids)}")
print(f"Number of images in folders (combined): {len(all_folder_ids)}")





