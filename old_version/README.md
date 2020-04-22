ME 759 Spring 2020 Final Project

Noah Van Der Weide

University of Wisconsin - Madison

Image upscaling using CUDA
--------------------------
The program uses OpenCV to take in any standard image format (.jpg, .png, .ppm, .bmp, etc.) and turn it into an array of unsigned chars. 

More will be added here as things progress..

This process is a blend of linear and nearest neighbor interpolation for standard image upscaling. Based on a threshold set by the user, two neighboring source pixels will fill in the destination pixels linearly or with nearest neighbor. This should ultimately have a similar output to something like bicubic interpolation, except with more accurate edges as there is no value overshoot. 
