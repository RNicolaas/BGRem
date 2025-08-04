
# BGRem: Background Removal for Astronomical Images

`BGRem` is a machine learning-based package designed specifically for the removal of background noise from astronomical images. It leverages a state-of-the-art diffusion model combined with an attention U-net architecture, ensuring precise and reliable background removal.


**Backbone of BGRem is an [Attention-UNet](https://arxiv.org/abs/1804.03999)**

![attention-u-net](../master/ExampleIms/Attention_Unet_schematic.png)


**An Example of BGRem on Optical Image**

![merlicht](../master/ExampleIms/Real_image_demonstration.png)  

## Features

- **Advanced Machine Learning Model**: Utilizes a diffusion model paired with an attention U-net, tailored for the nuanced needs of astronomical imaging.
- **Customizable Diffusion Steps**: Control the number of diffusion steps to balance between result precision and processing time.
- **Adaptive Normalization**: Fine-tune the normalization factor during pre-processing to cater to specific image characteristics.
- **Pixel Value Thresholding**: Set a threshold for minimal pixel value, enhancing the predictability of the model by avoiding unseen values during training.
- **Flexible Output Naming**: Customize the prefix of the output file name to streamline your workflow.

## Usage

Unpack the files in model.zip to the same map as bgrem.py, then run `BGRem` using the following command structure:

```
python3 bgrem.py -f <path_to_image> [-h <hdul location>] [-s <diffusion_steps>] [-n <normalisation_factor>] [-m <minimum_pixel_value>] [-p <output_prefix>]
```

### Parameters

- `-f` (MANDATORY): Path to the target image for background removal.
- `-h` (Optional): Location of image in hdul (default: 0). Integer value for image = hdul[h], to make sure the program finds the image and not a different object from the fits file.
- `-s` (Optional): Number of diffusion steps (default: 1). A higher number yields better results but increases processing time linearly.
- `-n` (Optional): Normalization factor for pre-processing (default: 0). Adjust to ensure the background noise's standard deviation is around 1. If set to 0, BGRem will estimate the normalisation factor itself.
- `-m` (Optional): Minimum pixel value in the image, excluding detector failures (default: 0). Helps maintain model predictability by avoiding unseen values during training.
- `-p` (Optional): Prefix for the output filename (default: 'BGRem_'). For example, a prefix of 'BGrem_' with a file name 'test_img.fits' will result in 'BGrem_test_img.fits'.

### Required Packages

Ensure you have the following packages installed, along with their dependencies:

- `numpy` (Tested with version 1.24.4)
- `tensorflow` (Tested with version 2.11.0)
- `astropy` (Tested with version 5.3.4)

---

`BGRem` is dedicated to providing astronomers with a powerful, intuitive tool for image processing, helping you focus on the universe's mysteries without the background noise.
