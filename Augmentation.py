import imgaug.augmenters as iaa
import cv2
import os
import glob

# Define the augmentation transformations
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.GaussianBlur(sigma=(0, 3.0)),  # gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # gaussian noise
    iaa.Multiply((0.8, 1.2)),  # brightness multiplier
    iaa.Affine(rotate=(-45, 45)),  # rotation
    iaa.ContrastNormalization((0.5, 2.0))  # contrast normalization
])

# Define the input directory and output directory
input_directory = 'Dataset/Yes'
output_directory = 'Dataset/Yes/Augmentation'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of image file paths in the input directory
image_paths = glob.glob(os.path.join(input_directory, '*.jpg'))

# Iterate through each image
for image_path in image_paths:
    # Load the input image
    input_image = cv2.imread(image_path)

    # Generate augmented images
    num_augmented_images = 30 # Number of augmented images to generate

    for i in range(num_augmented_images):
        # Perform augmentation on the input image
        augmented_image = seq.augment_image(input_image)

        # Get the filename from the image path
        filename = os.path.basename(image_path)

        # Construct the output path for the augmented image
        output_path = os.path.join(output_directory, f"augmented_{i}_{filename}")

        # Save the augmented image
        cv2.imwrite(output_path, augmented_image)

        print(f"Augmented image saved: {output_path}")