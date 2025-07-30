import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py

from tensorflow import keras
from keras import layers
from tensorflow.keras import backend as K

tf.compat.v1.enable_eager_execution()





# data
num_epochs = 100 
image_size = 256
cutout_size = [image_size,image_size]
plot_diffusion_steps = 10
validation_split = 0.1
empty_threshold = 350
normalisation_factor = 1e-2
randomise_noise_level = False
median_zero_noise = True
remove_below_zero = False

# sampling
min_signal_rate = 0
max_signal_rate = 0.98

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [16, 32, 64, 128, 256]
block_depth = 2

# optimization
batch_size = 32
ema = 0.999
start_learning_rate = 1e-3
learning_rate_decay = 0.92
weight_decay = 1e-4





def load_simulation_data(h5_file_path):
    all_simulations = {}

    with h5py.File(h5_file_path, 'r') as f:
        for sim_key in f.keys():
            simulation = {}

            # Load images
            simulation['full_image'] = np.array(f[sim_key]['full_image'])
            simulation['background_image'] = np.array(f[sim_key]['background_image'])

            # Load source info
            source_info_list = []
            source_info_grp = f[sim_key]['source_info']
            for source_key in source_info_grp.keys():
                source_info = {}
                for info_key in source_info_grp[source_key].keys():
                    source_info[info_key] = np.array(source_info_grp[source_key][info_key])
                source_info_list.append(source_info)
            simulation['source_info'] = source_info_list

            all_simulations[sim_key] = simulation

    return all_simulations

# Function to cut large image into small bits
def cut_image_in_pieces(image,cut_size):
    # Find the resolution of the image
    total_size = [len(image),len(image[0])]

    # Calculate maximum number of cuts of set size
    # in both dimensions
    max_y = total_size[1]//cut_size[1]
    max_x = total_size[0]//cut_size[0]

    # Store all cuts in array of images (the extra dimension at the end
    # is to make it work with the convolutional neural networks of the 
    # tensorflow keras package and only works for num_channels=1)
    data = np.zeros([max_x*max_y,cut_size[0],cut_size[1],1])
    for x in range(max_x):
        for y in range(max_y):
            data[max_y*x+y,:,:,0] = image[x*cut_size[0]:(x+1)*cut_size[0],y*cut_size[1]:(y+1)*cut_size[1]]
    
    # Return the array of images
    return(data)

def simulations_to_training_data(simulations,use_images=5,max_N=2000):
    X_train = [0]*use_images
    y_train = [0]*use_images
    for i in range(use_images):
        X_train[i] = cut_image_in_pieces(simulations['simulation_{}'.format(i+1)]['background_image'],cutout_size)
        y_train[i] = cut_image_in_pieces(simulations['simulation_{}'.format(i+1)]['full_image'],cutout_size)
        print(X_train[i].max())

        # Find the highest value in each cutout image
        # This will be used to remove empty images from the training data
        maximum_in_cutout = np.zeros(len(y_train[i]))
        for j in range(len(X_train[i])):
            maximum_in_cutout[j] = (y_train[i][j]-X_train[i][j]).max()

        # Only include images where the highest pixel value exceeds the empty threshold
        # This is to remove images with only background and no sources
        X_train[i] = X_train[i][maximum_in_cutout>empty_threshold] 
        y_train[i] = y_train[i][maximum_in_cutout>empty_threshold]

        y_train[i] = y_train[i]-X_train[i]

        if median_zero_noise:
            X_train[i] = X_train[i]-np.median(X_train[i])

    X_train = np.concatenate(X_train,axis=0)
    y_train = np.concatenate(y_train,axis=0)

    random_integers = np.arange(len(X_train))
    np.random.shuffle(random_integers)

    X_train = normalisation_factor*X_train[random_integers]
    y_train = normalisation_factor*y_train[random_integers]

    X_train = X_train[:max_N]
    y_train = y_train[:max_N]

    if randomise_noise_level:
        X_train = X_train*np.random.random([len(X_train),1,1,1])*4

    if remove_below_zero:
        below_zero_indices = np.where(X_train+y_train<0)
        X_train[below_zero_indices] = -y_train[below_zero_indices]

    return(X_train,y_train)

def plot_images(epoch=None, logs=None):
    global X_val, model, y_val
    predictions = model.predict(X_val,verbose=0)
    # Create the figure
    fig = plt.figure(figsize=(15, 3))

    # Get a random index
    i = np.random.randint(len(X_val))
    vmin = -0.0001
    vmax = 0.0005

    # Adds a subplot at the 1st position
    fig.add_subplot(1, 3, 2)

    # showing image
    plt.title('Predicted background')
    plt.imshow(predictions[i],cmap='gray',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.axis('off')

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 3, 1)

    # showing image
    plt.title('Real image')
    plt.imshow(X_val[i]+y_val[i],cmap='gray',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.axis('off')

    # Adds a subplot at the 3rd position
    fig.add_subplot(1, 3, 3)

    # showing image
    plt.title('Real Background image')
    # Get vmin and vmax to use zscale to make images easier to see with naked eye
    #vmin, vmax = ZScaleInterval().get_limits(y_val[i])
    plt.imshow(y_val[i],cmap='gray',vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.axis('off')

    plt.show()





# Get training data
h5_file_path = 'simulations/simulation_data_1-5.h5'  # Replace with your actual file path
X_train, y_train = simulations_to_training_data(load_simulation_data(h5_file_path),5,3000)
X_val = X_train[:int(validation_split*len(X_train))]
y_val = y_train[:int(validation_split*len(y_train))]
X_train = X_train[int(validation_split*len(X_train)):]
y_train = y_train[int(validation_split*len(y_train)):]
print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((y_train,X_train)).batch(batch_size,True)
val_dataset = tf.data.Dataset.from_tensor_slices((y_val,X_val)).batch(batch_size,True)






def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
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
   # Make sure all previous networks are gone, so memory doesn't get filled up with useless data
   K.clear_session()
    # inputs
   noisy_images = keras.Input(shape=(image_size, image_size, 1))
   noise_variances = keras.Input(shape=(1, 1, 1))

   e = layers.Lambda(sinusoidal_embedding)(noise_variances)
   e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

   x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
   x = layers.Concatenate()([x, e])
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(x, 32)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 64)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 128)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 256)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 512)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 256,True)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 128,True)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 64)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 32)
   # outputs
   outputs = layers.Conv2D(1, num_bands, padding="same", activation = "linear")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model([noisy_images, noise_variances], outputs, name="AU-Net")
   unet_model.summary()
   return unet_model





class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
#        self.normalizer = layers.Normalization()
        self.network = get_UNet_model(image_size, 1)
#        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
#        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

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
        network = self.network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = noisy_images - noise_rates * pred_noises

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
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def train_step(self, images):
        # unpack data
        real_images,input_images = images
        real_images = tf.cast(real_images, tf.float32)
        input_images = tf.cast(input_images, tf.float32)

        noises = input_images

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = real_images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(real_images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # unpack data
        real_images,input_images = images

        real_images = tf.cast(real_images, tf.float32)
        input_images = tf.cast(input_images, tf.float32)

        noises = input_images

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = real_images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(real_images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def generate_noise_removal(self, diffusion_steps, noisy_images, N=batch_size):      
        # noise -> images -> denormalized images
        generated_images = self.reverse_diffusion(noisy_images, diffusion_steps)
        return generated_images
    
    def plot_noise_removal(self, epoch=None, logs=None, diffusion_steps=plot_diffusion_steps, N=3):
        # Get training data from the 3 images
        noisy_images = X_val + y_val
        clean_images = y_val

        random_indices = np.random.randint(0,len(noisy_images),N)
        noisy_images = noisy_images[random_indices]
        clean_images = clean_images[random_indices]


        predictions = self.generate_noise_removal(diffusion_steps, noisy_images, N)

        for i in range(N):
            vmin = -100 * normalisation_factor
            vmax = 700 * normalisation_factor

            # Create the figure
            fig = plt.figure(figsize=(13, 4))

            # Adds a subplot at the 3rd position
            fig.add_subplot(1, 3, 3)

            # showing image
            plt.title('No background prediction')
            # Get vmin and vmax to use zscale to make images easier to see with naked eye
            #vmin, vmax = zscale().get_limits(predictions[i])
            plt.imshow(predictions[i],cmap='gray',vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.axis('off')

            # Adds a subplot at the 2nd position
            fig.add_subplot(1, 3, 2)

            # showing image
            plt.title('No background')
            # Get vmin and vmax to use zscale to make images easier to see with naked eye
            #vmin, vmax = zscale().get_limits(real_images[i])
            plt.imshow(clean_images[i],cmap='gray',vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.axis('off')

            # Adds a subplot at the 3rd position
            fig.add_subplot(1, 3, 1)

            # showing image
            plt.title('With background')
            # Get vmin and vmax to use zscale to make images easier to see with naked eye
            #vmin, vmax = zscale().get_limits(synthetic_images[i])
            plt.imshow(noisy_images[i],cmap='gray',vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.axis('off')

            # Show the images
            plt.savefig('noise_removal{}.png'.format(epoch))





# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=start_learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss

# save the best model based on the validation KID metric
checkpoint_path = "checkpoints/diffusion_model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_i_loss",
    mode="min",
    save_best_only=True,
)

def schedule(epoch):
   return(learning_rate_decay**epoch*start_learning_rate)

learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule, verbose=True)

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_noise_removal),
        checkpoint_callback,
        learning_rate_scheduler,
    ],
)