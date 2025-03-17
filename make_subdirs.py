import os
import pandas as pd
import shutil


def organize_images_into_classes(csv_file, image_dir, output_dir):
    """
    Organizes images into subdirectories based on class labels from a CSV file.

    Parameters:
    - csv_file: Path to the CSV file containing image filenames and their labels
    - image_dir: Directory containing all the images
    - output_dir: Directory where class subdirectories will be created

    Expected CSV format:
    filename,label
    img1.jpg,cat
    img2.jpg,dog
    ...
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Get unique classes
    unique_classes = df['label'].unique()

    # Create a subdirectory for each class
    for class_name in unique_classes:
        class_dir = os.path.join(output_dir, str(class_name))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # Copy each image to its corresponding class directory
    for index, row in df.iterrows():
        # Source image path
        src_path = os.path.join(image_dir, row['filename'])

        # Destination path in the appropriate class subdirectory
        dst_dir = os.path.join(output_dir, str(row['label']))
        dst_path = os.path.join(dst_dir, row['filename'])

        # Copy the image if it exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {row['filename']} to {dst_dir}")
        else:
            print(f"Warning: Image {src_path} not found")


# Example usage
# csv_file = "images/train_labels.csv"  # Path to your CSV file
# image_dir = "images/train"  # Directory with all your images
# output_dir = "junfan_subdirs"  # Where to create the organized directory structure
#
# organize_images_into_classes(csv_file, image_dir, output_dir)
#
# print("\nDirectory structure created successfully!")
# print(f"Your dataset is now organized in: {output_dir}")





