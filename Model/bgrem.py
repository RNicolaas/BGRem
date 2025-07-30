import math
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from astropy.io import fits
import argparse

tf.compat.v1.enable_eager_execution()


parser = argparse.ArgumentParser(description='Trains diffusion model on training data')
parser.add_argument('-f', type=str, help='file name')
parser.add_argument('-s', type=int, help='diffusion steps', default=1)
parser.add_argument('-i', type=int, help='location of image in hdul', default=0)
parser.add_argument('-n', type=float, help='normalisation factor', default=0)
parser.add_argument('-m', type=float, help='minimal pixel value that isn\'t detector failure', default=0)
parser.add_argument('-p', type=str, help='output file prefix', default='BGrem_')
args = parser.parse_args()

filename = args.f
diffusion_steps = args.s
prefix = args.p
minimal_pixel_value = args.m
hdul_nr = args.i

cutout_size = 256
use_size = 200
padding_width = (cutout_size-use_size)//2
extra_padding = [0,0]

# Diffusion model parameters
min_signal_rate = 0.01
max_signal_rate = 0.98
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [16, 32, 64, 128, 256]
block_depth = 2
normalisation_factor_diff = args.n
image_size = 256
cutout_size_diff = [image_size,image_size]
batch_size = 32


# Function to cut large image into small bits
def cut_image_in_pieces(image,cut_size,use_size):
    global extra_padding
    # Find the resolution of the image
    total_size = [len(image),len(image[0])]
    extra_padding = [use_size[0] - (total_size[0] % use_size[0]), use_size[1] - (total_size[1] % use_size[1])]

    # Calculate maximum number of cuts of set size
    # in both dimensions
    max_y = (total_size[1]+extra_padding[1])//use_size[1]
    max_x = (total_size[0]+extra_padding[0])//use_size[0]

    # Store all cuts in array of images (the extra dimension at the end
    # is to make it work with the convolutional neural networks of the 
    # tensorflow keras package and only works for num_channels=1)
    data = np.zeros([max_x*max_y,cut_size[0],cut_size[1],1])
    image = np.pad(image,[[padding_width,padding_width+extra_padding[0]],[padding_width,padding_width+extra_padding[0]]],mode='symmetric')
    for x in range(max_x):
        for y in range(max_y):
            data[max_y*x+y,:,:,0] = image[x*use_size[0]:x*use_size[0]+cut_size[0],y*use_size[1]:y*use_size[1]+cut_size[1]]
    
    # Return the array of images
    return(data)

# Function to reassemble image that was cut in pieces
def reasemble_image(data,original_img_size='unknown'):
    if original_img_size == 'unknown':
        # Find the resolution of the image
        total_size = [int(np.sqrt(len(data)))*use_size,int(np.sqrt(len(data)))*use_size] # Only works for square images
    else:
        total_size = original_img_size

    # Calculate maximum number of cuts of set size
    # in both dimensions
    max_y = math.ceil(total_size[1]/use_size)
    max_x = math.ceil(total_size[0]/use_size)

    reassembled_image = np.zeros([total_size[0]+extra_padding[0],total_size[1]+extra_padding[1]])
    for x in range(max_x):
        for y in range(max_y):
            reassembled_image[x*use_size:(x+1)*use_size,y*use_size:(y+1)*use_size] = data[max_y*x+y,padding_width:padding_width+use_size,padding_width:padding_width+use_size,0]

    return(reassembled_image)

# Function to substract the median pixel value from an image
def substract_median(array,ignore_minimum_value=False):
    if ignore_minimum_value:
        return(array-np.median(array[np.where(array != minimal_pixel_value*normalisation_factor_diff)]))
    else:
        return(array-np.median(array))
    
# Function to set all negative values to zero
def remove_below_zero(array):
    array[np.where(array<0)] = 0
    return(array)

# Function to open an image from a .fits file, this is a compressed version of .fits files
def open_fits_image(file_path):
    # Open the .fits file and import all the data
    hdul = fits.open(file_path)
    # Extract images from the full data, these are stored slightly differently compared to normal .fits files
    data = hdul[hdul_nr].data
    # Close the .fits file again
    hdul.close()

    # Return the image
    return(data)

# Function to open an image from a .fits.fz file, this is a compressed version of .fits files
def open_fitsfz_image(file_path):
    # Open the .fits file and import all the data
    hdul = fits.open(file_path)
    # Extract images from the full data, these are stored slightly differently compared to normal .fits files
    data = hdul[hdul_nr].data
    # Close the .fits file again
    hdul.close()

    # Return the image
    return(data)

# Remove background from an image
def image_bg_removal_diff(image, diff_model, diffusion_steps=3, normalisation_factor_diff=1e-2, lower_threshold=0, remove_negatives=False, individual_median_removal=False):
    # Pre-processing
    original_size = image.shape
    
    # Remove below lower threshold
    image[np.where(lower_threshold>image)] = lower_threshold

    # Normalize
    image *= normalisation_factor_diff

    if individual_median_removal:
        img_array = cut_image_in_pieces(image,[cutout_size,cutout_size],[use_size,use_size])
        for i in range(len(img_array)):
            img_array[i] = substract_median(img_array[i])
        if remove_negatives:
            for i in range(len(img_array)):
                img_array[i] = remove_below_zero(img_array[i])
    else:
        # Substract median value (centre noise around 0)
        image = substract_median(image,True)
        # Set negative values to 0
        if remove_negatives:
            image = remove_below_zero(image)
        # Cut the image in pieces
        img_array = cut_image_in_pieces(image,[cutout_size,cutout_size],[use_size,use_size])

    

    # Create array to put predictions
    reassembled_image_diff = 1e6*np.ones(img_array.shape)

    # Model noise removal

    # Remove noise with model
    for i in range(len(img_array)//10):
        reassembled_image_diff[10*i:10*(i+1)] = diff_model.generate_noise_removal(diffusion_steps,img_array[10*i:10*(i+1)])
        print(str(round((i+1)/len(img_array)*1000))+'%',end='\r')
    reassembled_image_diff[(len(img_array)//10)*10:] = diff_model.generate_noise_removal(diffusion_steps,img_array[(len(img_array)//10)*10:])
    print('100%')
    # Post-processing

    # Reassemble the image from the cutouts
    reassembled_image_diff = reasemble_image(reassembled_image_diff,original_size)
    # Remove extra padding
    reassembled_image_diff = reassembled_image_diff[:-1*extra_padding[0],:-1*extra_padding[1]]
    # Change all <0 values to 0
    reassembled_image_diff[np.where(reassembled_image_diff<0)] = 0
    # Denormalize
    reassembled_image_diff /= normalisation_factor_diff

    # Return background removed image
    return(reassembled_image_diff)

# Neural Network
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * np.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def double_conv_block(x, n_filters):
   # Symmetric padding
   x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "valid", activation = layers.LeakyReLU(alpha=0.2), kernel_initializer = "he_normal")(x)
   # Symmetric padding
   x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "valid", activation = layers.LeakyReLU(alpha=0.2), kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
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
   x = double_conv_block(x, n_filters)
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
   bottleneck = double_conv_block(p4, widths[4])
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

# Diffusion model
class DiffusionModel(keras.Model):
    def __init__(self, image_size):
        super().__init__()
        self.network = get_UNet_model(image_size, 1)

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (tf.cast(end_angle,tf.float32) - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        #if training:
        network = self.network
        #else:
        #    network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = noisy_images - noise_rates * pred_noises

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps,plot_diffusion_steps_now=False):
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
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate_noise_removal(self, diffusion_steps, noisy_images, N=batch_size):      
        # noise -> images -> denormalized images
        generated_images = self.reverse_diffusion(noisy_images, diffusion_steps)
        return generated_images

# create and compile the model
diff_model = DiffusionModel(image_size)
diff_model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=1e-3, weight_decay=1e-4
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss

# Load model from checkpoint
diff_model.load_weights('checkpoints/diffusion_model3').expect_partial()

if filename[-3:] == '.fz':
    real_image = open_fitsfz_image(filename)
elif filename[-5:] == '.fits':
    real_image = open_fits_image(filename)
else:
    print(filename)

if normalisation_factor_diff == 0:
    values = real_image.copy()[np.where(real_image>0)].flatten()
    values = np.sort(values)[:int(0.9*len(values))]
    std = np.std(values)
    normalisation_factor_diff = 1/std
    print("normalisation factor: " + str(normalisation_factor_diff))

real_image_background_removed = image_bg_removal_diff(real_image.copy(), diff_model, diffusion_steps, normalisation_factor_diff, minimal_pixel_value, False, True)

fits.writeto(prefix+filename,real_image_background_removed,overwrite=True)