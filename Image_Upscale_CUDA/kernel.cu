// Author: Noah Van Der Weide
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Upscale_CUDA.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
//#define THREADS_PER_BLOCK 64

__global__ void stretch_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int channels, unsigned int threshold);
__global__ void fill_CUDA(unsigned char* dst, int dst_width, int src_height, int channels, unsigned int threshold);

void upscale(unsigned char* src, unsigned char* dst, int src_height, int src_width, int dst_height, int dst_width, int channels, unsigned int threshold) {

	// initialize device variables
	unsigned char* dev_src, * dev_dst;

	// CUDA timing parameters
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms;

	// number of elements (if a picture has 3 channels, this is 3 * pixels)
	int dst_elements = dst_width * dst_height * channels;
	int src_elements = src_width * src_height * channels;

	// allocate memory in GPU
	cudaMalloc((void**)&dev_dst, dst_elements);
	cudaMalloc((void**)&dev_src, src_elements);

	// copy data from CPU to GPU
	cudaMemcpy(dev_dst, dst, dst_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_src, src, src_elements, cudaMemcpyHostToDevice);

	// start timer for performance evaluation
	cudaEventRecord(start);

	// call upscale function
	dim3 src_grid((src_width + 31) / 32, (src_height + 31) / 32);
	dim3 dst_grid((dst_width + 31) / 32, (src_height + 31) / 32);
	dim3 blocks(32, 32);

	stretch_CUDA << < src_grid, blocks >> > (dev_dst, dev_src, src_width, src_height, channels, threshold);
	cudaDeviceSynchronize();
	fill_CUDA << < dst_grid, blocks >> > (dev_dst, dst_width, src_height, channels, threshold);
	cudaDeviceSynchronize();

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	// copy data back from GPU to CPU
	cudaMemcpy(dst, dev_dst, dst_elements, cudaMemcpyDeviceToHost);

	// display time
	std::cout << "Upscale function finished in " << ms << " ms" << std::endl;

	// free GPU
	cudaFree(dev_dst);
	cudaFree(dev_src);
}

__global__ void stretch_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int channels, unsigned int threshold) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= src_width || y >= src_height)
		return;

	int dst_width = src_width * 3 - 2;
	int dst_index = ((x * 3) + (y * dst_width * 3)) * channels;
	int src_index = (x + y * src_width) * channels;

	int k;

	unsigned int diff = 0;
	unsigned int temp = 0;

	// check every channel for differences. If just one of the channels has a difference above the threshold, then apply nearest neighbor. 

	// horizontal
	for (k = 0; k < channels; k++) {
		temp = abs(src[src_index + channels + k] - src[src_index + k]); // difference between two color channels
		if (temp > diff)
			diff = temp;
	}

	if (diff > threshold) { // nearest neighbor
		for (k = 0; k < channels; k++) {
			dst[dst_index + k] = src[src_index + k];
			dst[dst_index + channels + k] = src[src_index + k];
			dst[dst_index + 2 * channels + k] = src[src_index + channels + k];
		}
	}
	else { // linear
		int step;
		for (k = 0; k < channels; k++) {
			step = (src[src_index + k] - src[src_index + channels + k]) / 3;
			dst[dst_index + k] = src[src_index + k];
			dst[dst_index + channels + k] = src[src_index + k] - step;
			dst[dst_index + 2 * channels + k] = src[src_index + k] - (2 * step);
		}
	}
}

__global__ void fill_CUDA(unsigned char* dst, int dst_width, int src_height, int channels, unsigned int threshold) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dst_width || y >= src_height)
		return;

	int dst_index = (x + (y * dst_width * 3)) * channels;

	int dst_stride = dst_width * channels;
	int k;

	unsigned int diff = 0;
	unsigned int temp = 0;

	// check every channel for differences. If just one of the channels has a difference above the threshold, then apply nearest neighbor. 

	for (k = 0; k < channels; k++) {
		temp = abs(dst[dst_index + k] - dst[dst_index + dst_stride * 3 + k]); // difference between two color channels
		if (temp > diff)
			diff = temp;
	}

	if (diff > threshold) { // nearest neighbor
		for (k = 0; k < channels; k++) {
			dst[dst_index + dst_stride + k] = dst[dst_index + k];
			dst[dst_index + dst_stride * 2 + k] = dst[dst_index + dst_stride * 3 + k];
		}
	}
	else { // linear
		int step;
		for (k = 0; k < channels; k++) {
			step = (dst[dst_index + k] - dst[dst_index + dst_stride * 3 + k]) / 3;
			dst[dst_index + dst_stride + k] = dst[dst_index + k] - step;
			dst[dst_index + dst_stride * 2 + k] = dst[dst_index + k] - (2 * step);
		}
	}
}


