// Author: Noah Van Der Weide
// 3/30/2020

#ifndef UPSCALE_CUH
#define UPSCALE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//void upscale(unsigned char* input_img, int height, int width, int channels, int threshold);
void upscale(unsigned char * dst, unsigned char * src, int src_height, int src_width, int src_channels, int threshold);
__global__ void upscale_CUDA(unsigned char* dst, unsigned char * src, int src_height, int src_width, int src_channels, int threshold);

__global__ void difference(int result, int Ax, int Ay, int Bx, int By, int stride, unsigned char * img);
__global__ void fill(int i, int j, int Ax, int Ay, int Bx, int By, int stride, int threshold, unsigned char * img, bool adjacent);

#endif
