import os
import cv2


def convert_images_to_black_and_white(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_image_path)
            if image is not None:
                # Convert the image to grayscale (black and white)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Save the grayscale image
                cv2.imwrite(output_image_path, gray_image)


# Example usage
input_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/images"
output_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/images/output_images_bw"
convert_images_to_black_and_white(input_folder, output_folder)
