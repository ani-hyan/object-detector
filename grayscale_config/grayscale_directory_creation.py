import os

base_path = '/data'
directories = ['train/images', 'valid/images', 'test/images']

for directory in directories:
    new_directory = os.path.join(base_path, f'grayscale_{directory}')
    os.makedirs(new_directory, exist_ok=True)
