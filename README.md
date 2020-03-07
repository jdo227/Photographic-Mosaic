# Photographic-Mosaic
This project creates photographic mosaic using images to replace sub-image in the main image.
## Steps
- Step 1: prepare image library and crop images into smaller size images
- Step 2: divide the main image into sub-images
- Step 3-1: for each sub-image, compare it with images in the library and find the image with best similarity using earth mover's distance (EMD) algorithm implemented in OpenCV
- Step 3-2: for each sub-image, compare it with images in the library and find the image with best similarity using 3-channel sum square distance method
- Step 4: replace the sub-image with the best-match in the library
- Step 5: place the replacements on canvas and create a new image
## How to run it?
```
make && ./Mosaic main_image pix_c source_lib target_lib
make && ./Mosaic main.jpg 50 lib_img_test/ lib_img_crop_test/ 1

```
pix_c: number of pixels in a colume of sub-image, e.g. 10.
source_lib: photo library with images before cropping.
target_lib: photo library cropped into designated sizes.

Notes: pix_c should not be too small to avoid black sub-images.
