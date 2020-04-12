// Author: Noah Van Der Weide
// 3/30/2020

#ifndef UPSCALE_CUH
#define UPSCALE_CUH

upscale_CUDA(unsigned char * input_img, int height, int width, int channels);
__global__ void upscale(int originalWidth, int originalHeight, int threshold, unsigned int *bmpOriginal, unsigned int *bmpNew);

int difference(int Ax, int Ay, int Bx, int By, int stride, unsigned int *img_original);
int fill(int i, int j, int Ax, int Ay, int Bx, int By, int stride, int threshold, unsigned int *img_original, bool adjacent);
