import os
import cv2
import numpy as np

class HandBlackener:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def blacken_hands(self, image_path):
        # Read the image
        image = cv2.imread(image_path)

        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the skin color in HSV space
        lower_skin = np.array([3, 40, 40], dtype=np.uint8)
        upper_skin = np.array([24, 255, 255], dtype=np.uint8)

        # Create a mask for the skin color
        skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

        # Create a black image
        black_image = np.zeros_like(image)

        # Use the mask to set the corresponding pixels in the original image to black
        image[skin_mask > 0] = black_image[skin_mask > 0]

        # Save the blackened image
        img_name = os.path.basename(image_path)
        output_image_path = os.path.join(self.output_folder, img_name)
        cv2.imwrite(output_image_path, image)

    def process_all_images(self):
        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Process each image in the input folder
        for filename in os.listdir(self.input_folder):
            input_image_path = os.path.join(self.input_folder, filename)

            # Check if the file is an image (you can add more extensions if needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Blacken the hands and save the result
                self.blacken_hands(input_image_path)

input_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/results/hand_detections"
blackened_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/results/blackened"
hand_blackener = HandBlackener(input_folder, blackened_folder)
hand_blackener.process_all_images()
