# PatchMatchInpainting

This project implements image inpainting using the PatchMatch algorithm. The PatchMatch algorithm is used to fill in missing regions of an image by finding similar patches from the known regions and copying them into the missing areas.

## Requirements

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- tqdm

You can install the required packages using pip:

```sh
pip install numpy opencv-python matplotlib tqdm
```

## Usage

### Single Resolution Inpainting

To perform inpainting using a single resolution, you can use the ```single_inpainting``` function. This function takes the image, mask, patch size, number of iterations, and a boolean flag for visualization as input.

```python
from patch_match import single_inpainting
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and mask
image = cv2.imread("test_data/yexinjia.png")
mask = cv2.imread("test_data/yexinjia.mask.png")
mask_color = [255, 0, 0]
mask = np.all(mask == mask_color, axis=-1)

# Perform inpainting
inpainted_image = single_inpainting(image, mask, patch_size=5, iterations=5, visual=True)

# Display the result
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Inpainted Image")
plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
plt.show()
```

### Pyramid Inpainting

To perform inpainting using a pyramid approach, you can use the ```pyramid_inpainting``` function. This function takes the image, mask, patch size, and a boolean flag for visualization as input.

```python
from patch_match import pyramid_inpainting
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and mask
image = cv2.imread("test_data/yexinjia.png")
mask = cv2.imread("test_data/yexinjia.mask.png")
mask_color = [255, 0, 0]
mask = np.all(mask == mask_color, axis=-1)

# Perform inpainting
inpainted_image = pyramid_inpainting(image, mask, patch_size=5, visual=True)

# Display the result
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Inpainted Image")
plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
plt.show()
```

### Functions

#### PatchMatch

The `PatchMatch` class implements the PatchMatch algorithm. It includes the following methods:

- `__init__(self, source_image, mask, patch_size)`: Initializes the `PatchMatch` object with the source image, mask, and patch size.
- `coord_in_image(self, x, y)`: Checks if coordinates are within image bounds.
- `patch_in_image(self, x, y)`: Checks if patch is within image bounds.
- `compute_patch_distance(self, patch_image1, patch_coords1, patch_image2, patch_coords2)`: Computes the sum of squared differences between two patches.
- `random_initialize_nnf(self)`: Initializes the nearest neighbor field (NNF) with random valid coordinates.
- `paint_matched(self, target)`: Paints the target image using the NNF.
- `run(self, iterations=5, visual=False, random_nnf=True, nnf=None)`: Performs inpainting using PatchMatch.

#### Additional Functions

- `downsample(image, new_height, new_width)`: Downsamples the image and mask to `new_height` x `new_width` using mean pooling.
- `upsample(nnf, new_height, new_width)`: Upsamples the NNF to `new_height` x `new_width` using bilinear interpolation.
- `pyramid_inpainting(image, mask, patch_size=5, visual=False)`: Performs inpainting using a pyramid approach.
- `single_inpainting(image, mask, patch_size=5, iterations=5, visual=False)`: Performs inpainting using a single resolution.
