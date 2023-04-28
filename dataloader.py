import os
import cv2
import numpy as np
import tensorflow as tf

batch_size = 8
patch_size = (256, 256)
num_patches = 12  # number of patches per image
num_channels = 3  # assuming RGB images

# define function to load and preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    img = img.astype(np.float32) / 255.0  # normalize pixel values to [0, 1]
    return img

# define function to generate patches from image
def generate_patches(img1, img2, num_patches, patch_size):
    patches1 = []
    patches2 = []
    h, w, c = img1.shape
    for i in range(num_patches):
        x = np.random.randint(0, w - patch_size[0] + 1)
        y = np.random.randint(0, h - patch_size[1] + 1)
        patch1 = img1[y:y+patch_size[1], x:x+patch_size[0], :]
        patch2 = img2[y:y+patch_size[1], x:x+patch_size[0], :]
        patches1.append(patch1)
        patches2.append(patch2)
    return patches1, patches2

# define function to preprocess patches
def preprocess_patches(patches):
    return np.array(patches)

# define dataset loader
def create_dataset(input_file, output_file):
    # read input and output paths from text files
    with open(input_file, 'r') as f:
        input_paths = f.read().splitlines()
    with open(output_file, 'r') as f:
        output_paths = f.read().splitlines()

    inimages = []
    gtimages = []
    for input_path, output_path in zip(input_paths, output_paths):
        blur_img = load_image(input_path)
        sharp_img = load_image(output_path)
        inpatches, gtpatches = generate_patches(blur_img, sharp_img, num_patches, patch_size)            
        inpatches = preprocess_patches(inpatches)
        gtpatches = preprocess_patches(gtpatches)
        inimages.extend(inpatches)
        gtimages.extend(gtpatches)

    inimages = np.array(inimages)
    gtimages = np.array(gtimages)
    dataset = tf.data.Dataset.from_tensor_slices((inimages, gtimages))
    dataset = dataset.shuffle(len(inimages))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset