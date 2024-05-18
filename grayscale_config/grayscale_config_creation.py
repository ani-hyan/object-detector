data_config = """
path: /Users/ahakobyan/Desktop/capstone/Object-Detector/data
train: ../grayscale_train/images
val: ../grayscale_valid/images
test: ../grayscale_test/images

names:
  0: 'pointing_gesture'
"""

with open("../grayscale_config.yaml", "w") as f:
    f.write(data_config)