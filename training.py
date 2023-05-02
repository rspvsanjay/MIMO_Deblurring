import re
import os
import time 
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from networks import MIMO_Network
from dataloader import create_dataset

loss_level1 = tf.keras.metrics.Mean('loss_level1', dtype=tf.float32)
loss_level2 = tf.keras.metrics.Mean('loss_level2', dtype=tf.float32)
loss_level3 = tf.keras.metrics.Mean('loss_level3', dtype=tf.float32)

fft_loss_level1 = tf.keras.metrics.Mean('fft_loss_level1', dtype=tf.float32)
fft_loss_level2 = tf.keras.metrics.Mean('fft_loss_level2', dtype=tf.float32)
fft_loss_level3 = tf.keras.metrics.Mean('fft_loss_level3', dtype=tf.float32)
loss_at_level1_epoch = tf.keras.metrics.Mean('loss_at_level1_epoch', dtype=tf.float32)

# set up the training loop
@tf.function
def train_step(inputs, targets):
    targets1 = targets
    targets2 = tf.image.resize(targets1, [tf.shape(targets1)[1]//2, tf.shape(targets1)[2]//2])
    targets3 = tf.image.resize(targets2, [tf.shape(targets2)[1]//2, tf.shape(targets2)[2]//2])

    loss1 = 0
    loss2 = 0
    loss3 = 0

    fft1_loss = 0
    fft2_loss = 0
    fft3_loss = 0
    with tf.GradientTape() as tape3, tf.GradientTape() as tape2, tf.GradientTape() as tape1:
        # forward pass
        pred3, pred2, pred1 = model(inputs, training=True)
        
        # calculate loss of level3
        loss3 = mae_loss_fn(targets3, pred3)
        targets3_fft = tf.signal.fft(tf.cast(targets3, tf.complex64))
        preds3_fft = tf.signal.fft(tf.cast(pred3, tf.complex64))
        # calculate FFT loss
        fft3_loss = tf.reduce_mean(tf.abs(targets3_fft - preds3_fft))
        loss3 = loss3 + 0.1 * fft3_loss

        # calculate loss of level2
        loss2 = mae_loss_fn(targets2, pred2)
        targets2_fft = tf.signal.fft(tf.cast(targets2, tf.complex64))
        preds2_fft = tf.signal.fft(tf.cast(pred2, tf.complex64))
        # calculate FFT loss
        fft2_loss = tf.reduce_mean(tf.abs(targets2_fft - preds2_fft))
        loss2 = (loss2 + loss3)/2 + 0.1 * fft2_loss

        # calculate loss of level1
        loss1 = mae_loss_fn(targets1, pred1)
        targets1_fft = tf.signal.fft(tf.cast(targets1, tf.complex64))
        preds1_fft = tf.signal.fft(tf.cast(pred1, tf.complex64))
        # calculate FFT loss
        fft1_loss = tf.reduce_mean(tf.abs(targets1_fft - preds1_fft))
        loss1 = (loss1 + loss2 + loss3)/3 + 0.1 * fft1_loss

    grads3 = tape3.gradient(loss3, model.get_layer("convo1k3s1f3l3").trainable_variables) # calculate gradients at level3
    optimizer.apply_gradients(zip(grads3, model.get_layer("convo1k3s1f3l3").trainable_variables)) # update weights
    loss_level3(loss3)
    fft_loss_level3(fft3_loss)

    grads2 = tape2.gradient(loss2, model.get_layer("convo1k3s1f3l2").trainable_variables) # calculate gradients at level2
    optimizer.apply_gradients(zip(grads2, model.get_layer("convo1k3s1f3l2").trainable_variables)) # update weights
    loss_level2(loss2)
    fft_loss_level2(fft2_loss)
    
    grads1 = tape1.gradient(loss1, model.trainable_variables) # calculate gradients at level1
    optimizer.apply_gradients(zip(grads1, model.trainable_variables)) # update weights
    loss_level1(loss1)
    fft_loss_level1(fft1_loss)

    return loss1

# read input and output paths from text files
input_file = "/content/drive/MyDrive/StartCode3/input_train.txt"
output_file = "/content/drive/MyDrive/StartCode3/output_train.txt"
with open(input_file, 'r') as f:
    input_paths = f.read().splitlines()
with open(output_file, 'r') as f:
    output_paths = f.read().splitlines()

index = []
for num1 in range(len(output_paths)):
    index.append(num1)

def indexing(index, lengthOfBatch):
    random.shuffle(index)
    length = len(output_paths)/lengthOfBatch # 25 is batch size for a epoch
    index2 =  [0] * round(length)
    for num1 in range(round(length)):
        index3 = []
        for num2 in range(num1*lengthOfBatch, (num1+1)*lengthOfBatch):
            if num2<len(index):
                index3.append(index[num2])
        index2[num1] = index3
    return index2
index4 = indexing(index, 25)
input_batch_path = []
output_batch_path = []
for num1 in range(len(index4[0])):
    input_batch_path.append(input_paths[index4[0][num1]])
    output_batch_path.append(output_paths[index4[0][num1]])
#load data
train_data = create_dataset(input_batch_path, output_batch_path)
numberofbatch = len(index4)

#load Networks
model = MIMO_Network()
# define the loss function
mae_loss_fn = tf.keras.losses.MeanAbsoluteError()

# define the optimizer with a learning rate schedule
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=500, decay_rate=0.5, staircase=True)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
# Use the legacy optimizer tf.keras.optimizers.legacy.Adam instead of the tf.keras.optimizers.Adam. 
# The legacy optimizer does not require the list of variables to be registered separately.

# compile the model with loss and metric functions
model.compile(optimizer=optimizer, loss=mae_loss_fn, metrics=[tf.keras.metrics.MeanAbsoluteError()])

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/content/drive/MyDrive/StartCode3/logs/gradient_tape1/' + current_time + '/train'
train_summary_writer1 = tf.summary.create_file_writer(train_log_dir)
train_log_dir = '/content/drive/MyDrive/StartCode3/logs/gradient_tape2/' + current_time + '/train'
train_summary_writer2 = tf.summary.create_file_writer(train_log_dir)
# set up the checkpoint
checkpoint_dir = '/content/drive/MyDrive/StartCode3/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# restore the latest checkpoint (if it exists)
epoch_to_restore = 0
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    match = re.search(r'ckpt_(\d+)', latest_checkpoint) # extract the epoch number from the checkpoint file name
    if match:
        epoch_to_restore = int(match.group(1))
    print("epoch_to_restore: ", epoch_to_restore)
    checkpoint.restore(latest_checkpoint)

# set up the TensorBoard callback
log_dir = "/content/drive/MyDrive/StartCode3/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

num_epochs = 4000
batch_count = 0
for epoch in range(epoch_to_restore, num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    start_time_epoch = time.time()  # record start time
    loss = 0
    for number1 in range(numberofbatch-1):
        start_time = time.time()  # record start time
        numberofminibatch = len(train_data)
        mini_batch = 0
        for step, (inputs, targets) in enumerate(train_data):
            start_timeb = time.time()  # record start time
            loss = train_step(inputs, targets) 
            end_timeb = time.time()  # record end time 
        end_time = time.time()  # record end time 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Epoch: {} in progress, Loss = {}, Batch Number: {}/{}, Time taken for a batch = {:.2f}s, Total Number of Mini-batch: {}, Time taken for a mini-batch = {:.2f}s, Current Time: {} ".format(epoch+1, loss.numpy(),  number1+1, numberofbatch, end_time - start_time, numberofminibatch, end_timeb - start_timeb, current_time))
        with train_summary_writer1.as_default():
            tf.summary.scalar('loss_level3', loss_level3.result(), step=batch_count)
            tf.summary.scalar('loss_level2', loss_level2.result(), step=batch_count)
            tf.summary.scalar('loss_level1', loss_level1.result(), step=batch_count)
            tf.summary.scalar('fft_loss_level3', fft_loss_level3.result(), step=batch_count)
            tf.summary.scalar('fft_loss_level2', fft_loss_level2.result(), step=batch_count)
            tf.summary.scalar('fft_loss_level1', fft_loss_level1.result(), step=batch_count)
        batch_count = batch_count + 1
        # Reset metrics every batch
        fft_loss_level3.reset_states()
        fft_loss_level2.reset_states()
        fft_loss_level1.reset_states()
        loss_level3.reset_states()
        loss_level2.reset_states()
        loss_level1.reset_states()
        # create the list of paths
        input_batch_path = []
        output_batch_path = []
        for num1 in range(len(index4[number1+1])):
            input_batch_path.append(input_paths[index4[number1+1][num1]])
            output_batch_path.append(output_paths[index4[number1+1][num1]])
        #load data
        train_data = create_dataset(input_batch_path, output_batch_path)
    # add loss to TensorBoard
    loss_at_level1_epoch(loss)
    with train_summary_writer2.as_default():
        tf.summary.scalar('loss_at_level1_epoch', loss_at_level1_epoch.result(), step=epoch)
    # Reset metrics every epoch
    loss_at_level1_epoch.reset_states()
    # save checkpoint every 30 epoch
    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix.format(epoch=epoch))
        print("checkpoint_prefix.format(epoch=epoch): ", checkpoint_prefix.format(epoch=epoch))  
    end_time_epoch = time.time()  # record end time  
    print("Epoch {} is completed: Loss = {}, Batch Number: {}/{}, Time taken to complete a epoch = {:.2f}s, Current Time: {} ".format(epoch+1, loss.numpy(),  numberofbatch, numberofbatch, end_time_epoch - start_time_epoch, current_time))
    random.shuffle(index)
    index4 = indexing(index, 25)
    input_batch_path = []
    output_batch_path = []
    for num1 in range(len(index4[0])):
        input_batch_path.append(input_paths[index4[0][num1]])
        output_batch_path.append(output_paths[index4[0][num1]])
    #load data
    train_data = create_dataset(input_batch_path, output_batch_path)
    print("data loaded")
