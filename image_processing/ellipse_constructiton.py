import os
import cv2
import numpy as np

class EllipseDrawer:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def draw_ellipses_and_vectors(self, image_path):
        # Read the original image
        image = cv2.imread(image_path)

        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the skin color in HSV space
        lower_skin = np.array([3, 40, 40], dtype=np.uint8)
        upper_skin = np.array([24, 255, 255], dtype=np.uint8)

        # Create a mask for the skin color
        skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

        # Find contours in the skin mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all detected contours
        for contour in contours:
            # Get the bounding ellipse around the contour
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(image, ellipse, (255, 0, 0), 1)  # Draw the ellipse in blue with thinner lines

                # Calculate the center and orientation of the ellipse
                (center, axes, angle) = ellipse
                center = (int(center[0]), int(center[1]))

                # Determine the major and minor axis lengths
                major_axis_length = max(axes) / 2
                minor_axis_length = min(axes) / 2

                # Increase the length of the minor axis for drawing purposes
                length_factor = 10  # Further increase this factor to make the minor axis longer
                extended_minor_axis_length = minor_axis_length * length_factor

                # Calculate the angle in radians
                angle_rad = np.deg2rad(angle)

                # Calculate the end points of the extended minor axis vector
                minor_axis_end1 = (
                    int(center[0] + extended_minor_axis_length * np.cos(angle_rad + np.pi / 2)),
                    int(center[1] + extended_minor_axis_length * np.sin(angle_rad + np.pi / 2))
                )
                minor_axis_end2 = (
                    int(center[0] - extended_minor_axis_length * np.cos(angle_rad + np.pi / 2)),
                    int(center[1] - extended_minor_axis_length * np.sin(angle_rad + np.pi / 2))
                )

                # Draw the extended minor axis vector (L2)
                cv2.line(image, minor_axis_end1, minor_axis_end2, (255, 255, 0), 1)  # Draw the minor axis in yellow for reference

        # Save the image with ellipses and direction vectors
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
                # Draw ellipses and vectors on the original image and save the result
                self.draw_ellipses_and_vectors(input_image_path)

input_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/results/hand_detections"
ellipses_folder = "/Users/ahakobyan/Desktop/capstone/Object-Detector/results/ellipses"
ellipse_drawer = EllipseDrawer(input_folder, ellipses_folder)
ellipse_drawer.process_all_images()
