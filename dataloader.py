import os
import cv2
import random
import numpy as np

class DataLoader():
    def __init__(self, patch_size=(256, 256), batch_size=16, num_patches=12):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_patches = num_patches

    # define function to load and preprocess image
    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
        img = img.astype(np.float32) / 255.0  # normalize pixel values to [0, 1]
        return img

    # define function to generate patches from image
    def generate_patches(self, img1, img2, num_patches, patch_size):
        patches1 = []
        patches2 = []
        h, w, c = img1.shape
        for i in range(num_patches):
            x = np.random.randint(0, w - patch_size[0] + 1)
            y = np.random.randint(0, h - patch_size[1] + 1)
            patch1 = img1[y:y + patch_size[1], x:x + patch_size[0], :]
            patch2 = img2[y:y + patch_size[1], x:x + patch_size[0], :]
            patches1.append(patch1)
            patches2.append(patch2)
            patch1 = cv2.flip(patch1, 1)
            patch2 = cv2.flip(patch2, 1)
            patches1.append(patch1)
            patches2.append(patch2)
            # print("Number of patches2: ", len(patches2))
        return patches1, patches2

    # define patch collector
    def collect_patches(self, input_paths, output_paths):
        inimages = []
        gtimages = []
        for input_path, output_path in zip(input_paths, output_paths):
            blur_img = self.load_image(input_path)
            sharp_img = self.load_image(output_path)
            inpatches, gtpatches = self.generate_patches(blur_img, sharp_img, self.num_patches, self.patch_size)
            inimages.extend(inpatches)
            gtimages.extend(gtpatches)
            # print("Number of patches after each image: ", len(inimages))
        return inimages, gtimages

    def load_batches(self, input_paths, output_paths):
        # print("load_batches called")
        inpatchimages, gtpatchimages = self.collect_patches(input_paths, output_paths)
        # print("Total Number of patches after each image: ", len(inpatchimages))
        index = []
        for num1 in range(len(inpatchimages)):
            index.append(num1)
        random.shuffle(index)

        length = len(inpatchimages) // self.batch_size
        index2 = [0] * length
        for num1 in range(length):
            index3 = []
            for num2 in range(num1 * self.batch_size, (num1 + 1) * self.batch_size):
                if num2 < len(index):
                    index3.append(index[num2])
            index2[num1] = index3
        for i in range(len(index2)):
            input_batch = []
            output_batch = []
            for j in range(len(index2[i])):
                input_batch.append(inpatchimages[index2[i][j]])
                output_batch.append(gtpatchimages[index2[i][j]])
            yield input_batch, output_batch, length

input_file = "/content/drive/MyDrive/StartCode3/input_train_100.txt"
output_file = "/content/drive/MyDrive/StartCode3/output_train_100.txt"
with open(input_file, 'r') as f:
    input_paths = f.read().splitlines()
with open(output_file, 'r') as f:
    output_paths = f.read().splitlines()
