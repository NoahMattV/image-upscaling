
#ifndef UPSCALE_CUH
#define UPSCALE_CUH

__global__ void upscale(int originalWidth, int originalHeight, int threshold, unsigned int *bmpOriginal, unsigned int *bmpNew);
