// Author: Noah Van Der Weide
// 3/30/2020

#ifndef _Upscale_CUDA_
#define _Upscale_CUDA_

void upscale(unsigned char* src, unsigned char* dst, int src_height, int src_width, int dst_height, int dst_width, int channels, unsigned char threshold);

#endif