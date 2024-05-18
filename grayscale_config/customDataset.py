import cv2
import os
import shutil

base_path = '/data'
directories = ['train', 'valid', 'test']

def copy_and_convert_to_grayscale(image_path, save_path):
    # Copy the original image first
    shutil.copy(image_path, save_path)
    # Convert the copied image to grayscale
    image = cv2.imread(save_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, gray_image)

# Copy images to new directories, convert to grayscale, and copy the labels
for directory in directories:
    old_image_path = os.path.join(base_path, directory, 'images')
    old_label_path = os.path.join(base_path, directory, 'labels')

    new_image_path = os.path.join(base_path, f'grayscale_{directory}/images')
    new_label_path = os.path.join(base_path, f'grayscale_{directory}/labels')

    os.makedirs(new_image_path, exist_ok=True)
    os.makedirs(new_label_path, exist_ok=True)

    for filename in os.listdir(old_image_path):
        if filename.endswith(".jpg") or filename.endswith(".webp") or filename.endswith(".jpeg") or filename.endswith(".png"):  # Add other image extensions if needed
            image_path = os.path.join(old_image_path, filename)
            save_image_path = os.path.join(new_image_path, filename)
            copy_and_convert_to_grayscale(image_path, save_image_path)

            # Copy the corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            old_label_file = os.path.join(old_label_path, label_filename)
            new_label_file = os.path.join(new_label_path, label_filename)
            if os.path.exists(old_label_file):
                shutil.copy(old_label_file, new_label_file)

print("Conversion to grayscale, and label copying completed.")