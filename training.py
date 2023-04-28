import re
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from networks import MIMO_Network
from dataloader import create_dataset
#load data
train_data = create_dataset("/content/drive/MyDrive/StartCoding/input_train20.txt", "/content/drive/MyDrive/StartCoding/output_train20.txt")
#load Networks
model = MIMO_Network()
# define the loss function
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# define the optimizer
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.legacy.Adam()
# Use the legacy optimizer tf.keras.optimizers.legacy.Adam instead of the tf.keras.optimizers.Adam. 
# The legacy optimizer does not require the list of variables to be registered separately.
# set up the checkpoint
checkpoint_dir = '/content/drive/MyDrive/StartCode3/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# restore the latest checkpoint (if it exists)
epoch_to_restore = 0
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    # extract the epoch number from the checkpoint file name
    checkpoint_prefix = os.path.join(checkpoint_dir, "")
    match = re.search(r'ckpt_(\d+)', latest_checkpoint)
    if match:
        epoch_to_restore = int(match.group(1))
    print("epoch_to_restore: ", epoch_to_restore)
    checkpoint.restore(latest_checkpoint)

# set up the TensorBoard callback
log_dir = "/content/drive/MyDrive/StartCode3/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# set up the training loop
@tf.function
def train_step(inputs, targets):
    targets1 = targets
    targets2 = tf.image.resize(targets1, [tf.shape(targets1)[1]//2, tf.shape(targets1)[2]//2])
    targets3 = tf.image.resize(targets2, [tf.shape(targets2)[1]//2, tf.shape(targets2)[2]//2])
    with tf.GradientTape() as tape3, tf.GradientTape() as tape2, tf.GradientTape() as tape1:
        # forward pass
        pred3, pred2, pred1 = model(inputs, training=True)
        
        # calculate loss of level3
        loss3 = mse_loss_fn(targets3, pred3)
        targets3_fft = tf.signal.fft(tf.cast(targets3, tf.complex64))
        preds3_fft = tf.signal.fft(tf.cast(pred3, tf.complex64))
        # calculate FFT loss
        fft3_loss = tf.reduce_mean(tf.abs(targets3_fft - preds3_fft))
        loss3 = loss3 + 0.1 * fft3_loss

        # calculate loss of level2
        loss2 = mse_loss_fn(targets2, pred2)
        targets2_fft = tf.signal.fft(tf.cast(targets2, tf.complex64))
        preds2_fft = tf.signal.fft(tf.cast(pred2, tf.complex64))
        # calculate FFT loss
        fft2_loss = tf.reduce_mean(tf.abs(targets2_fft - preds2_fft))
        loss2 = (loss2 + loss3)/2 + 0.1 * fft2_loss

        # calculate loss of level1
        loss1 = mse_loss_fn(targets1, pred1)
        targets1_fft = tf.signal.fft(tf.cast(targets1, tf.complex64))
        preds1_fft = tf.signal.fft(tf.cast(pred1, tf.complex64))
        # calculate FFT loss
        fft1_loss = tf.reduce_mean(tf.abs(targets1_fft - preds1_fft))
        loss1 = (loss1 + loss2 + loss3)/3 + 0.1 * fft1_loss

    # calculate gradients of level3
    grads3 = tape3.gradient(loss3, model.get_layer("convo1k3s1f3l3").trainable_variables)
    optimizer.apply_gradients(zip(grads3, model.get_layer("convo1k3s1f3l3").trainable_variables)) # update weights

    # calculate gradients of level2
    grads2 = tape2.gradient(loss2, model.get_layer("convo1k3s1f3l2").trainable_variables)
    optimizer.apply_gradients(zip(grads2, model.get_layer("convo1k3s1f3l2").trainable_variables)) # update weights
    
    # calculate gradients of level1
    grads1 = tape1.gradient(loss1, model.trainable_variables)
    optimizer.apply_gradients(zip(grads1, model.trainable_variables)) # update weights

    return loss1

num_epochs = 20
for epoch in range(epoch_to_restore, num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    for step, (inputs, targets) in enumerate(train_data):
        loss = train_step(inputs, targets)        
        # print loss every 10 steps
        if (step + 1) % 4 == 0:
            print("Step {}: Loss = {}".format(step, loss.numpy()))        
        # save checkpoint every 100 steps
        if (step + 1) % 4 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix.format(epoch=epoch))    
    # add loss to TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar('loss', loss, step=step)