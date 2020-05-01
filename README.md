# Image Upscaling with CUDA
ME 759 Final Project - Upscaling an image with optional thresholds

University of Wisconsin - Madison

Noah Van Der Weide


# What it does
It takes a user specified image (placed in the same folder as the .exe) and increased the dimensions by (dim*3-2). 
The empty pixels are then filled with either a nearest neighbor or linear interpolation based on whether or not the
values of any of the channels exceed the user-defined threshold. 

If threshold is set to 0, then the program simply fills every pixel using nearest neighbor.
Threshold at 255 is bilinear (or linear)

![]()

# Why this method?
There are several ways to interpolate an image when resizing. One of the most common is bicubic, which takes pixels 
up to two spaces away and generates a curve based on their values which is used to assign the empty pixels with a value.
This works well for most applications, however it can cause unintended changes if there are small 'islands' of a specific
color. The interpolated values may over/undershoot drastically based on the size of the island. 
This is poor for dense checker-boarded images. 

My method of using a threshold allows the user to tweak how the interpolation is handled to generate a more accurate 
upscaled image in some cases. 

# My specs
- NVIDIA GeForce GTX 1060 with 6 GB VRAM
- 16 GB RAM
- Intel Core i5-7500 CPU @ 3.40 GHz, 4 cores.

# Software used
- CUDA 10.2
- Visual Studio 2019
- OpenCV2 4.3.0
- Windows 10
