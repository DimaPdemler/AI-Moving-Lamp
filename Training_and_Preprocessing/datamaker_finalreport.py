import os
import cv2
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set target resolution for processed images
W, H = 320, 240  

# Supported image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}  

def generate(input_path, output_path):
    # Process each file in the input directory
    for frame_file in tqdm(os.listdir(input_path)):
        try:
            # Check if the file is an image based on its extension
            if os.path.splitext(frame_file)[1].lower() in image_extensions:
                # Read image in grayscale for simplicity and reduced file size
                img = cv2.imread(os.path.join(input_path, frame_file), cv2.IMREAD_GRAYSCALE)

                # Rotate image if its height is greater than its width
                if img.shape[0] > img.shape[1]:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # Resize image to predefined dimensions
                img = cv2.resize(img, (W, H))

                # Write the processed image to the output path
                cv2.imwrite(os.path.join(output_path, frame_file), img)

                # Prepare filenames for flipped images
                fname, fext = os.path.splitext(frame_file)
                save1 = output_path + fname + '-0' + fext
                save2 = output_path + fname + '-1' + fext

                # Create and save flipped versions of the image
                cv2.imwrite(save1, cv2.flip(img, 0))  # Vertical flip
                cv2.imwrite(save2, cv2.flip(img, 1))  # Horizontal flip
        except Exception as e:
            # Handle any exceptions during processing
            print('Failed:', frame_file, '; Error:', e)

# Base directory containing image datasets
base_path = '/Users/dimademler/Desktop/UCSD/2023-2024/Quarter_1/Physics_124/Temp_picture_data/'

# Different sets of input directories for processing different datasets
# Uncomment the relevant line for the desired dataset
input_names = ['Background_validation_2', 'Background_validation']
# input_names = ['Signal_Dec9', 'Signal_Nov30', 'Signal2_nov16']
# input_names = ['Background_Dec9', 'Background_nov30', 'Background_rand_nov26', 'Background_nov26', 'Background_nov16']
# input_names = ['Signal_validation_2', 'Signal_validation']

# Output path for processed images
output_path = '/Users/dimademler/Desktop/UCSD/2023-2024/Quarter_1/Physics_124/Temp_picture_data/Dec10_validation_preprocessed/background/'

# Process each specified input directory
for input_name in input_names:
    input_path = os.path.join(base_path, input_name)
    generate(input_path, output_path)

# Example of processing a single dataset
# input_path = '/Users/dimademler/Desktop/UCSD/2023-2024/Quarter_1/Physics_124/Temp_picture_data/Background_validation/'
# generate(input_path, output_path)
