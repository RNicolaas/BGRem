'''
written with a similar structure to train_bgrem.py
'''
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle

# from astropy.io import fits
# from astropy.visualization import ZScaleInterval as zscale

from tensorflow import keras
from keras import layers
from tensorflow.keras import backend as K

# tf.compat.v1.enable_eager_execution()


# data
num_epochs = 150 # train for at least 50 epochs for good results
image_size = 64
cutout_size = [image_size,image_size]
plot_diffusion_steps = 10

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.98

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [16, 16, 16, 32, 32]
block_depth = 2

# optimization
batch_size = 16
ema = 0.999
start_learning_rate = 8e-4
learning_rate_decay = 0.94
weight_decay = 1e-4
checkpoint_path = "/path_to_ckpt/checkpoints/"

# training data
# nr_images = int(5e3)
validation_split = 0.15
normalisation_factor = 1e-2
im_path = '/path_to_data/test_im_iem_psr_bll_fsrq_pwn1_2_patch768_tbin1/'
test_diffusion_steps = 3





###################
# added sb 21052025
###################

def get_split_file_lists(path, N_train=4000, N_test=300, seed=42):
    image_names = sorted(os.listdir(path))  # ensure consistency
    np.random.seed(seed)
    np.random.shuffle(image_names)

    train_files = image_names[:N_train]
    test_files = image_names[N_train:N_train + N_test]
    
    return train_files, test_files

#################

def training_data_diff_model(file_list, path=im_path, image_size=image_size):
    N = len(file_list)
    # image_names = os.listdir(path)
    # image_names = np.flip(image_names,axis=0)
    background_images = np.zeros((N, image_size, image_size, 1))
    groundtruth_images = np.zeros((N, image_size, image_size, 1))
    for i, fname in enumerate(file_list):
        image = np.load(os.path.join(path, fname))
        groundtruth_images[i] = image[1] + image[2] + image[3] + image[4]
        background_images[i] = image[0]
    
    # Ten years
    groundtruth_images *= 10
    background_images *= 10

    groundtruth_images[np.where(groundtruth_images<0)[0]] = 0
    background_images[np.where(background_images<0)[0]] = 0

    # make observations # switched off to focus only on background (IEM); No Poisson
    # background_images += np.random.poisson(background_images) - background_images
    # groundtruth_images += np.random.poisson(groundtruth_images) - groundtruth_images

    # Normalisation
    background_images *= normalisation_factor
    groundtruth_images *= normalisation_factor

    print('Background images shape: '+str(background_images.shape)+' ; Ground truth images shape: '+str(groundtruth_images.shape))
    return (background_images, groundtruth_images)

def test_data(file_list, path=im_path, image_size=image_size):
    N = len(file_list)
    images = np.zeros((N, image_size, image_size, 1))
    for i, fname  in enumerate(file_list):
        image = np.load(os.path.join(path, fname))
        images[i] = image[0] + image[1] + image[2] + image[3] + image[4]
    
    # Ten years
    images *= 10

    images[np.where(images<0)[0]] = 0

    # make observations
    # images += np.random.poisson(images) - images

    # Normalisation
    images *= normalisation_factor

    print('Images shape: '+str(images.shape))
    return (images)





#######################
# train test split
######################

# Get training data
# X_train, y_train = training_data_diff_model(nr_images)

# Get split
train_files, test_files = get_split_file_lists(im_path, N_train=12000, N_test=600, seed=42)
print ('check train files: ', train_files[0:5])
X_train, y_train = training_data_diff_model(train_files, im_path, image_size)

X_val = X_train[:int(validation_split*len(X_train))]
y_val = y_train[:int(validation_split*len(y_train))]
# X_train = X_train[int(validation_split*len(X_train)):]
# y_train = y_train[int(validation_split*len(y_train)):]

X_val, y_val = shuffle(X_val, y_val, random_state=2)

print('check shapes of train + val: ',  X_train.shape,y_train.shape,X_val.shape,y_val.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((y_train, X_train)).batch(batch_size, True)
val_dataset = tf.data.Dataset.from_tensor_slices((y_val, X_val)).batch(batch_size, True)


test_images = test_data(test_files, im_path, image_size)


#####################
# added: random image pairs plotting to check
#####################

# Pick 4 random indices
indices = np.random.choice(len(X_train), size=4, replace=False)

# Create figure
fig, axes = plt.subplots(4, 2, figsize=(8, 12))  # 4 rows, 2 columns (X, Y)

for i, idx in enumerate(indices):
    noisy = X_train[idx, :, :, 0] / normalisation_factor
    clean = y_train[idx, :, :, 0] / normalisation_factor

    # Left: Noisy input
    axes[i, 0].imshow(noisy, cmap='inferno')
    axes[i, 0].set_title("Noisy Input (X)")
    axes[i, 0].axis('off')

    # Right: Clean target
    axes[i, 1].imshow(clean, cmap='inferno')
    axes[i, 1].set_title("Clean Source (Y)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('./random_train_test_img.png', dpi=200)
#plt.show()

#####################
# start  building network
####################

# Sine embedding
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(tf.linspace(tf.math.log(embedding_min_frequency), 
                                     tf.math.log(embedding_max_frequency), 
                                     embedding_dims // 2,))
    angular_speeds = 2.0 * np.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), 
                            tf.cos(angular_speeds * x)], axis=3)
    return embeddings

###################
# added 12/5/25 -sb
# compatible for tf 2
###################
class DoubleConvBlock(keras.layers.Layer):
    def __init__(self, n_filters, **kwargs):
        super(DoubleConvBlock, self).__init__(**kwargs)
        self.n_filters = n_filters   
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.n_filters, 3, padding='valid', 
                                   activation=layers.LeakyReLU(alpha=0.2), 
                                   kernel_initializer="he_normal")
        self.conv2 = layers.Conv2D(self.n_filters, 3, padding='valid', 
                                   activation=layers.LeakyReLU(alpha=0.2), 
                                   kernel_initializer='he_normal')    
    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        x = self.conv1(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        x = self.conv2(x)

        return x


def downsample_block(x, n_filters):
   # changed 12/5/25 
   # f = double_conv_block(x, n_filters)
   conv_block = DoubleConvBlock(n_filters)
   f = conv_block(x) 
   p = layers.MaxPool2D(2)(f)
#   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters,has_attention=False):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # Attention layer
   if has_attention:
      conv_features = attention_gate(x, conv_features, n_filters)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
#   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   conv_block = DoubleConvBlock(n_filters) 
   x = conv_block(x)
   return x

def attention_gate(g,s,n_filters):
    # Correct number of filters and batch normalisation
    Wg = layers.Conv2D(n_filters,1,padding='same')(g)
    Wg = layers.BatchNormalization()(Wg)
    # Correct number of filters and batch normalisation
    Ws = layers.Conv2D(n_filters,1,padding='same')(s)
    Ws = layers.BatchNormalization()(Ws) 
    # Add together and ReLU
    out = layers.Activation('relu')(Wg + Ws) 
    out = layers.Conv2D(n_filters,1,padding='same',activation='sigmoid')(out)
    # Multiply with skip connection
    return(out * s)

def get_UNet_model(img_size, num_bands):
    # inputs
   noisy_images = keras.Input(shape=(img_size, img_size, 1))
   noise_variances = keras.Input(shape=(1, 1, 1))

   e = layers.Lambda(sinusoidal_embedding)(noise_variances)
   e = layers.UpSampling2D(size=img_size, interpolation="nearest")(e)

   x = layers.Conv2D(embedding_dims, kernel_size=1)(noisy_images)
   x = layers.Concatenate()([x, e])
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(x, widths[0])
   # 2 - downsample
   f2, p2 = downsample_block(p1, widths[1])
   # 3 - downsample
   f3, p3 = downsample_block(p2, widths[2])
   # 4 - downsample
   f4, p4 = downsample_block(p3, widths[3])
   # 5 - bottleneck
   # bottleneck = double_conv_block(p4, widths[4])
   # added 12/5/25
   bottleneck_conv = DoubleConvBlock(256)
   bottleneck = bottleneck_conv(p4) 
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, widths[3],True)
   # 7 - upsample
   u7 = upsample_block(u6, f3, widths[2],True)
   # 8 - upsample
   u8 = upsample_block(u7, f2, widths[1])
   # 9 - upsample
   u9 = upsample_block(u8, f1, widths[0])
   # outputs
   outputs = layers.Conv2D(1, num_bands, padding="same", activation = "linear")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model([noisy_images, noise_variances], outputs, name="AU-Net")
   return unet_model


#### define model

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        self.network = get_UNet_model(image_size, 1)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.psnr_metric_tracker = keras.metrics.Mean(name="psnr") 
        # added this on 20062025, for a separate tracker  

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.psnr_metric_tracker]

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.math.acos(max_signal_rate)
        end_angle = tf.math.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (tf.cast(end_angle,tf.float32) - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.math.cos(diffusion_angles)
        noise_rates = tf.math.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        network = self.network
        # print ('within def denoise')
        # noise_variances = tf.reshape(noise_rates ** 2, (-1, 1, 1, 1))  # (B, 1, 1, 1)
        # noise_variances = tf.cast(noise_variances, dtype=tf.float32) # other complains that noise_rates not a tensor
        noise_variances = tf.convert_to_tensor(noise_rates ** 2, dtype=tf.float32)
        noise_variances = tf.reshape(noise_variances, [-1, 1, 1, 1])


        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_variances], training=training)
        pred_images = noisy_images - noise_rates * pred_noises
        
        ######## for debugging 

        # print ('noisy ims_shape: ', noisy_images.shape)
        # print ('noise rate shape: ', noise_rates.shape)
        # print ('noise vars shape: ', noise_variances.shape)
        # print ('type noise vars + images: ', type(noise_variances), type(noisy_images))
        ################

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False)
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (pred_images + next_noise_rates * pred_noises)
            # this new noisy image will be used in the next step

        return pred_images

    def train_step(self, images):
        # unpack data
        real_images,input_images = images
        real_images = tf.cast(real_images, tf.float32)
        input_images = tf.cast(input_images, tf.float32)

        noises = input_images

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = real_images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, 
                                                    signal_rates, training=True)

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(real_images, pred_images)  # only used as metric
            psnr_vals = tf.image.psnr(real_images, pred_images, max_val=1.0)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.psnr_metric_tracker.update_state(psnr_vals)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # unpack data
        real_images,input_images = images

        real_images = tf.cast(real_images, tf.float32)
        input_images = tf.cast(input_images, tf.float32)

        noises = input_images

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), 
                                            minval=0.0, maxval=1.0)
        
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = real_images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(real_images, pred_images)
        psnr_vals  = tf.image.psnr(real_images, pred_images, max_val=1.0)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.psnr_metric_tracker.update_state(psnr_vals)

        return {m.name: m.result() for m in self.metrics}

    def generate_noise_removal(self, diffusion_steps, noisy_images, N=batch_size):      
        # noise -> images -> denormalized images
        generated_images = self.reverse_diffusion(noisy_images, diffusion_steps)
        return generated_images
    



    def plot_callback(val_dataset, diffusion_steps=10, N=3, normalisation_factor=1):
    # this one plots different noisy and noise removed images for every epoch
        def plot_noise_removal(epoch=None, logs=None,  ):
        # Create an iterator from the validation dataset
            # val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size)

            val_iterator = iter(val_dataset)
        
            # Extract N batches from the dataset (assuming each batch has at least one image)
            noisy_images, clean_images = [], []
            for _ in range(N):
                batch_noisy, batch_clean = next(val_iterator)
                noisy_images.append(batch_noisy[0])  # Assuming batch size is greater than 0
                clean_images.append(batch_clean[0])
        
            # noisy_images = np.array(noisy_images)
            noisy_images = tf.convert_to_tensor(np.array(noisy_images), dtype=tf.float32)
            clean_images = np.array(clean_images)

            full_images = noisy_images + clean_images

            

        
            # Generate predictions
            predictions = model.generate_noise_removal(diffusion_steps, full_images, N)

            fig, axes = plt.subplots(N, 3, figsize=(13, 12))

            # vmin = -100 * normalisation_factor
            # vmax = 700 * normalisation_factor

        
            for i in range(N):
                

                print ('check shapes in plot: ', np.shape(noisy_images[i]), np.shape(clean_images[i]))
                print ('check max, min of the noisy and clean: ', np.max(noisy_images[i]), 
                       np.min(noisy_images[i]), np.max(clean_images[i]), np.min(clean_images[i]))
            
                # Create the figure
                # fig = plt.figure(figsize=(13, 4))
            
                # Adds a subplot at the 1st position
                # ax1 = fig.add_subplot(1, 3, 1)
                # ax1.imshow(full_images[i], cmap='inferno', )
                axes[i, 0].imshow(full_images[i], cmap='inferno')
                # ax1.title.set_text('With Background')
                axes[i, 0].set_title(f'Sample {i+1}: With Bkg. (Input)')
                # ax1.axis('off')
                axes[i, 0].axis('off')
            
                # Adds a subplot at the 2nd position
                # ax2 = fig.add_subplot(1, 3, 2)
                # ax2.imshow(clean_images[i], cmap='inferno', )
                # axes[i, 1].imshow(clean_images[i], cmap='inferno')
                axes[i, 1].imshow(noisy_images[i], cmap='inferno')
                # ax2.title.set_text('Only Background')
                # axes[i, 1].set_title(f'Sample {i+1}: Only Bkg.')
                axes[i, 1].set_title(f'Sample {i+1}: No Bkg.(Target)')

                # ax2.axis('off')
                axes[i, 1].axis('off')
                
                # Adds a subplot at the 3rd position
                # ax3 = fig.add_subplot(1, 3, 3)
                # ax3.imshow(predictions[i], cmap='inferno', )
                axes[i, 2].imshow(predictions[i], cmap='inferno')
                # ax3.title.set_text('No Background Prediction')
                axes[i, 2].set_title(f'Sample {i+1}: Prediction')
                # ax3.axis('off')
                axes[i, 2].axis('off')
                
                # Show the images
            plt.tight_layout()
            plt.savefig(f'./pred_im_denoise_noPoisson/diffusion_pred_example_noPoi_embed_dim{embedding_dims}_step{epoch}.png', dpi=200)
            print (f'saved noise removal images for epoch {epoch}')
        return plot_noise_removal

########### final train

# Diffusion model

# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=start_learning_rate, 
                                                  weight_decay=weight_decay), 
              loss=tf.keras.losses.MeanAbsoluteError())
# pixelwise mean squared error is used as loss

# Load model from checkpoint
#model.load_weights("checkpoints/diffusion_model_gammaray2")

# save the best model based on the validation KID metric
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path + f'bgrem_trial_LAT_NoPoisson_embed{embedding_dims}.weights.h5', 
                                                         save_weights_only=True, 
                                                         monitor="val_i_loss", 
                                                         mode="min", 
                                                         save_best_only=True,)

def schedule(epoch):
   return(learning_rate_decay**epoch*start_learning_rate)

## ls schedule
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=True)

## early stop
# Early stopping: stop training if val_i_loss doesn't improve for 10 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_i_loss", patience=10, 
                                                  mode="min", restore_best_weights=True, verbose=1)

# run training and plot generated images periodically
history2 = model.fit(train_dataset, epochs=num_epochs, 
                     validation_data=val_dataset, 
                     callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=DiffusionModel.plot_callback(val_dataset)), 
                                checkpoint_callback, learning_rate_scheduler, early_stopping],)

########
# plot loss
########

fig = plt.figure(figsize=(8, 5))
fig.add_subplot(121)
# plt.title('Loss + PSNR during training for initialisation with random weights')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(history2.history['i_loss'],label='Train loss')
plt.plot(history2.history['val_i_loss'],label='Valid. loss')
plt.yscale('log')
plt.legend(fontsize=11)
fig.add_subplot(122)
plt.ylabel('PSNR')
plt.xlabel('Epoch')
plt.plot(history2.history['psnr'],label='Train PSNR')
plt.plot(history2.history['val_psnr'],label='Valid. PSNR')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(checkpoint_path + 'check_loss_Trial_NoPoisson.png', dpi=200)
# plt.show()


###################
# version testing
##################
# tensorflow ---> 2.18.0
# keras      ---> 3.6.0
# numpy      ---> 1.26.4
# matplotlib ---> 3.8.4