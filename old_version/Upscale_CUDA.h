// Author: Noah Van Der Weide
// 3/30/2020

#ifndef _Upscale_CUDA_
#define _Upscale_CUDA_

void Upscale_CUDA(unsigned char* src, unsigned char * dst, int src_height, int src_width, int dst_height, int dst_width, int channels, unsigned char threshold);
//void upscale(unsigned char* input_img, int height, int width, int channels, int threshold);
//void upscale(unsigned char* dst, unsigned char* src, int src_height, int src_width, int src_channels, int threshold);
//__global__ void upscale_CUDA(unsigned char* dev_dst, unsigned char* dev_src, int src_width, int src_height, int src_channels, unsigned char threshold);
//__global__ void upscale(unsigned char threshold);
#endif
