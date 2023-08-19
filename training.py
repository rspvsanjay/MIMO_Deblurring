from sklearn.model_selection import train_test_split
from networks import MIMO_Deblur
from dataloader import DataLoader
import tensorflow as tf
from PIL import Image
import numpy as np
import threading
import datetime
import random
import json
import os
import re

# Load configuration from JSON file
with open('/content/drive/MyDrive/StartCode3/Code3/config.json', 'r') as config_file:
    config = json.load(config_file)

# Data paths
input_file = config['data_paths']['input_file']
output_file = config['data_paths']['output_file']
test_input_dir = config['data_paths']['test_input_dir']
test_output_dir = config['data_paths']['test_output_dir']

# Model parameters
num_filters = config['model_params']['num_filters']
batch_size = config['model_params']['batch_size']
epochs = config['model_params']['epochs']

# Training parameters
learning_rate = config['training_params']['optimizer_lr']
fft_loss_weight = config['training_params']['fft_loss_weight']
num_indexes = config['training_params']['num_indexes']

# File paths
checkpoint_dir = config['file_paths']['checkpoint_dir']
log_dir = config['file_paths']['log_dir']
deblurred_dir = config['file_paths']['deblurred_dir']

# Now you can use these variables in your code as before.

with open(input_file, 'r') as f:
    input_paths = f.read().splitlines()

with open(output_file, 'r') as f:
    output_paths = f.read().splitlines()

# Shuffle the input and output paths in the same order
combined_paths = list(zip(input_paths, output_paths))
random.shuffle(combined_paths)
shuffled_input_paths, shuffled_output_paths = zip(*combined_paths)

# Split the data into training and testing sets (90% training, 10% testing)
input_paths_train, input_paths_test, output_paths_train, output_paths_test = train_test_split(
    shuffled_input_paths, shuffled_output_paths, test_size=0.10, random_state=42)

# Instantiate the AutoEncoder model
autoencoder = MIMO_Deblur()
dataloader = DataLoader()

# Define the optimizers
optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer3 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def deblur_image(img_path, model):
    # Load the blurry image from the given path
    img = Image.open(img_path)
    # Convert the image to numpy array and normalize it to [0, 1]
    img_array = np.array(img) / 255.0
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    # Perform deblurring using the model
    deblurred_array = model.predict(img_array)
    # Convert the deblurred numpy array back to an image and scale it to [0, 255]
    deblurred_img_array = np.squeeze(deblurred_array[2]) * 255.0
    deblurred_img_array = np.clip(deblurred_img_array, 0, 255).astype(np.uint8)
    deblurred_img = Image.fromarray(deblurred_img_array)
    return deblurred_img

def save_deblurred_images(original_img_path, deblurred_img, gt_path, epoch):
    original_dir, original_filename = os.path.split(original_img_path)
    os.makedirs(deblurred_dir, exist_ok=True)

    # Save the blurred image
    original_img = Image.open(original_img_path)
    original_filename1 = "blur_" + original_filename
    original_img_path = os.path.join(deblurred_dir, original_filename1)
    original_img.save(original_img_path)

    # Save the deblurred image
    deblurred_filename = "deblur_" + f'epoch{epoch:04d}_' + original_filename
    deblurred_img_path = os.path.join(deblurred_dir, deblurred_filename)
    deblurred_img.save(deblurred_img_path)

    # Save the Sharp image
    original_img = Image.open(gt_path)
    original_dir, original_filename = os.path.split(gt_path)
    original_filename2 = "sharp_"+ original_filename
    original_img_path = os.path.join(deblurred_dir, original_filename2)
    original_img.save(original_img_path)


def fft_loss(y_true, y_pred):
    # Compute the FFT of the ground-truth and predicted images
    y_true_fft = tf.signal.fft2d(tf.cast(y_true, tf.complex64))
    y_pred_fft = tf.signal.fft2d(tf.cast(y_pred, tf.complex64))
    # Compute the absolute difference between the FFTs
    fft_loss = tf.reduce_mean(tf.abs(y_true_fft - y_pred_fft))
    return fft_loss


losses1 = []
losses2 = []
losses3 = []
fft_losses1 = []
fft_losses2 = []
fft_losses3 = []

global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)  # Create a global step variable

@tf.function
def train_step(inputs, targets):
    # Resize targets by half using tf.image.resize
    targets_half = tf.image.resize(targets, [tf.shape(targets)[1] // 2, tf.shape(targets)[2] // 2])
    targets_fourth = tf.image.resize(targets, [tf.shape(targets)[1] // 4, tf.shape(targets)[2] // 4])
    # print(targets_half.shape)
    # print(targets_fourth.shape)
    with tf.GradientTape() as tape3, tf.GradientTape() as tape2, tf.GradientTape() as tape1:
        # Forward pass through the AutoEncoder model
        fake_B = autoencoder(inputs, training=True)

        # Compute loss and gradients for fake_B[0]
        loss3 = tf.reduce_mean(tf.abs(fake_B[0] - targets_fourth))  # Using targets[2] for img_B_half2
        loss_fft = fft_loss(targets_fourth, fake_B[0])
        loss3 = loss3 + loss_fft * 0.01
        losses3.append(loss3)
        fft_losses3.append(loss_fft)

        # Compute loss and gradients for fake_B[1]
        loss2 = tf.reduce_mean(tf.abs(fake_B[1] - targets_half))  # Using targets[1] for img_B_half1
        loss_fft = fft_loss(targets_half, fake_B[1])
        loss2 = loss2 + loss_fft * 0.01
        loss2 = (loss2 + loss3)/2
        losses2.append(loss2)
        fft_losses2.append(loss_fft)

        # Compute loss and gradients for fake_B[2]
        loss1 = tf.reduce_mean(tf.abs(fake_B[2] - targets))  # Using targets[0] for img_B
        loss_fft = fft_loss(targets, fake_B[2])
        loss1 = loss1 + loss_fft * 0.01
        loss1 = (loss1 + loss2 + loss3)/3
        losses1.append(loss1)
        fft_losses1.append(loss_fft)

    # Compute gradients and update weights for each output separately
    gradients3 = tape3.gradient(loss3, autoencoder.get_layer("convo1k3s1f3l3").trainable_variables) 
    optimizer3.apply_gradients(zip(gradients3, autoencoder.get_layer("convo1k3s1f3l3").trainable_variables))

    gradients2 = tape2.gradient(loss2, autoencoder.get_layer("convo1k3s1f3l2").trainable_variables)
    optimizer2.apply_gradients(zip(gradients2, autoencoder.get_layer("convo1k3s1f3l2").trainable_variables))

    gradients1 = tape1.gradient(loss1, autoencoder.trainable_variables)
    optimizer1.apply_gradients(zip(gradients1, autoencoder.trainable_variables))

    return loss1, loss2, loss3

# Define a directory to save the model checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the global step variable
global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# Create a checkpoint instance
checkpoint = tf.train.Checkpoint(model=autoencoder, optimizer1=optimizer1, optimizer2=optimizer2, optimizer3=optimizer3, global_step=global_step)

# Function to extract epoch number from checkpoint filename
def extract_epoch_from_checkpoint(checkpoint_filename):
    match = re.search(r'epoch_(\d+)_step', checkpoint_filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Try to restore from the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Restoring from {latest_checkpoint}")
    checkpoint.restore(latest_checkpoint)
    restored_epoch = extract_epoch_from_checkpoint(latest_checkpoint)
    if restored_epoch is not None:
        start_epoch = restored_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Restore the global_step value from the checkpoint
    global_step.assign(int(latest_checkpoint.split('_')[-1].split('.')[0]))  # Extract and assign the step number
    print("global_step number: ", global_step.numpy())
else:
    start_epoch = 0

# Define the summary writer
summary_writer = tf.summary.create_file_writer(log_dir)
summary_writer1 = tf.summary.create_file_writer(log_dir)

# num_indexes = 40
paths_per_index = len(input_paths_train) // num_indexes

# Initialize ds3 as None globally
ds2 = None
# Define a function to load ds3
def load_ds2(inpath, outpath):
    global ds2
    ds2 = dataloader.load_batches(inpath, outpath)
    print("Ds2 loaded")

for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch}/{epochs}")
    # Shuffle the input and output paths in the same order
    combined_paths = list(zip(input_paths_train, output_paths_train))
    random.shuffle(combined_paths)
    input_paths_train, output_paths_train = zip(*combined_paths)

    for i in range(num_indexes-1):
        print("i: ", i)
        if i == 0:
            start_idx = i * paths_per_index
            end_idx = (i + 1) * paths_per_index
            inpath1 = input_paths_train[start_idx:end_idx]
            outpath1 = output_paths_train[start_idx:end_idx]
            print("inpath1 length: ", len(inpath1))
            print("outpath1 length: ", len(outpath1))
            ds1 = dataloader.load_batches(inpath1, outpath1)
            print("Ds1 loaded")

        start_idx = (i + 1) * paths_per_index
        end_idx = (i + 2) * paths_per_index
        inpath2 = input_paths_train[start_idx:end_idx]
        outpath2 = output_paths_train[start_idx:end_idx]
        print("inpath2 length: ", len(inpath2))
        print("outpath2 length: ", len(outpath2))

        # Create a thread to load ds3 and pass inpath3 and outpath3 as arguments
        load_ds2_thread = threading.Thread(target=load_ds2, args=(inpath2, outpath2))
        # Start the thread
        load_ds2_thread.start()

        for batch_i, (input_batch, output_batch, length) in enumerate(ds1):
            # Convert input_batch and output_batch to numpy arrays
            input_batch = np.array(input_batch)
            output_batch = np.array(output_batch)

            # Train the model on the current batch using the custom train_step function
            loss1, loss2, loss3 = train_step(input_batch, output_batch)

            # Get the current time
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Print or log the losses with the current time
            print(f"{current_time} - Epoch: {epoch}/{epochs} - Batch: {length*i + batch_i+1:03d}/{length*num_indexes} - Losses: L1 {loss1:.4f}, L2 {loss2:.4f}, L3 {loss3:.4f}")

            # Update the global step
            global_step.assign_add(1)

            # Write the losses to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("loss1", loss1, step=global_step.numpy())
                tf.summary.scalar("loss2", loss2, step=global_step.numpy())
                tf.summary.scalar("loss3", loss3, step=global_step.numpy())
        # Wait for the thread to finish
        load_ds2_thread.join()
        ds1 = ds2
        if i%8==0:
            # Save the model checkpoint with .ckpt extension
            checkpoint_filename = f"epoch_{epoch:04d}_step_{global_step.numpy():08d}_data_index_{i:02d}.ckpt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            checkpoint.save(checkpoint_path)

    with summary_writer1.as_default():
        tf.summary.scalar("loss1perEpoch", loss1, step=epoch)
        tf.summary.scalar("loss2perEpoch", loss2, step=epoch)
        tf.summary.scalar("loss3perEpoch", loss3, step=epoch)
    # Save the model checkpoint with .ckpt extension
    checkpoint_filename = f"epoch_{epoch:04d}_step_{global_step.numpy():08d}.ckpt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    checkpoint.save(checkpoint_path)

    # Call deblur_image and save_deblurred_images for test images
    for i in range(len(input_paths_test)):
        input_img_path = input_paths_test[i]
        output_img_path = output_paths_test[i]
        
        # Call deblur_image function
        deblurred_img = deblur_image(input_img_path, autoencoder)
        
        # Call save_deblurred_images function
        save_deblurred_images(input_img_path, deblurred_img, output_img_path, epoch)
