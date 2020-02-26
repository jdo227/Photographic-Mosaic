# Photographic-Mosaic
This project creates photographic mosaic using images to replace sub-image in the main image.
# Step: prepare image library and crop images into smaller size images
# Step: divide the main image into sub-images
# Step: for each sub-image, compare it with images in the library and find the image with best similarity using earth mover's distance (EMD) algorithm implemented in OpenCV
# Step: replace the sub-image with the best-match in the library
# Step: place the replacements on canvas and create a new image