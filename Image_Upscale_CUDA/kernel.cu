// Author: Noah Van Der Weide
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Upscale_CUDA.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#define THREADS_PER_BLOCK 64

__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels, unsigned char threshold);
__global__ void stretch_CUDA(unsigned char* dst, unsigned char* srci, int src_width, int src_height, int src_channels);

void upscale(unsigned char* src, unsigned char* dst, int src_height, int src_width, int dst_height, int dst_width, int channels, unsigned char threshold) {
    // initialize device variables
    unsigned char* dev_src, * dev_dst;

    // number of elements (if a picture has 3 channels, this is 3 * pixels)
    int dst_elements = dst_width * dst_height * channels;
    int src_elements = src_width * src_height * channels;

    // number of bytes each image will take
    int dst_size = dst_elements * sizeof(unsigned char);
    int src_size = src_elements * sizeof(unsigned char);

    // number of blocks to call in kernel. Max threads per block is usually 1024
    //int blocks = (src_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // allocate memory in GPU
    cudaMalloc((void**)&dev_dst, dst_elements);
    cudaMalloc((void**)&dev_src, src_elements);
    // used for shared memory if eventually implemented
    //cudaMallocManaged(&dst, dst_elements);
    //cudaMallocManaged(&src, src_elements);

    // copy data from CPU to GPU
    cudaMemcpy(dev_dst, dst, dst_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_src, src, src_elements, cudaMemcpyHostToDevice);

    // start timer for performance evaluation
    //cudaEventRecord(start);

    // call upscale function
    //upscale_CUDA<<<blocks, THREADS_PER_BLOCK>>>  (dev_dst, dev_src, src_elements, src_width, src_height, threshold); // <<<blocks, threads per block, shared mem>>>
    dim3 grid((src_width + 31)/32, (src_height + 31)/32);
    dim3 blocks(32, 32);
    //dim3 grid(src_width, src_height); // use with <<<grid, 1>>>
    upscale_CUDA << <grid, blocks >> > (dev_dst, dev_src, src_width, src_height, channels, threshold);
    //stretch_CUDA <<<grid, 1>>>(dev_dst, dev_src, src_width, src_height, channels);
    //stretch_CUDA << <grid, blocks >> > (dev_dst, dev_src, src_width, src_height, channels);
    cudaDeviceSynchronize();

    // end timer
    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    */
    // copy data back from GPU to CPU
    cudaMemcpy(dst, dev_dst, dst_elements, cudaMemcpyDeviceToHost);
    //cudaMemcpy(src, dev_src, dst_elements, cudaMemcpyDeviceToHost); // might not need this

    // free GPU
    cudaFree(dev_dst);
    cudaFree(dev_src);

}

__global__ void stretch_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //int x = blockIdx.x;
    //int y = blockIdx.y;
    
    if (x >= src_width || y >= src_height) 
        return;

    int dst_width = src_width * 3 - 2;
    //int dst_index = ((x * 3) + (y * dst_width*3)) * src_channels;
    int dst_index = ((x * 3) + (y * dst_width)) * src_channels;
    int src_index = (x + y * src_width) * src_channels;
    //int dst_index = src_index * 3;

    for (int k = 0; k < src_channels; k++) {
        // transfer known src values to dst
        // to access different channels, the number of elements of the src/dst image must be added to the respective array index.
        dst[dst_index + k] = src[src_index + k];
        dst[dst_index + src_channels + k] = src[src_index + k];
        dst[dst_index + 2*src_channels + k] = src[src_index + k];
        
    }
}
/*
__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels, unsigned char threshold) {

    int x = blockIdx.x;
    int y = blockIdx.y;

    int dst_width = src_width * 3 - 2;

    int src_stride = src_width * src_channels;
    int dst_stride = dst_width * src_channels;

    if (x >= src_width || y >= src_height)
        return;

    //int dst_index = (y * 21 + x * 3);
    int dst_index = ((x * 3) + (y * dst_width));
    int src_index = (x + y * src_width);

    // all channels for a pixel are grouped together. To access an adjacent pixel, you must add by the number of channels.
    for (int k = 0; k < src_channels; k++) {
        

        // transfer known src values to dst
        // to access different channels, the number of elements of the src/dst image must be added to the respective array index.
        dst[dst_index + k] = src[src_index + k];
        dst[dst_index + src_channels + k] = src[src_index + k];
        dst[dst_index + 2*src_channels + k] = src[src_index + k];
    }
    __syncthreads();
}
*/


__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels, unsigned char threshold) {

    // not using shared memory right now
    // there is 48 KB of shared memory available.
    // images are typically more than that, so I'll have to think about how it could be implemented
    //extern __shared__ unsigned char pic[];

  //int pixel = blockIdx.x * blockdim.x + threadIdx.x;



    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //int x = blockIdx.x;
    //int y = blockIdx.y;
    // not relevant to code function, but shows how a thread could access a pixel in every channel.
    // pixel values are from 0 to 255.
    //for (int k = 0; k < channels; k++){
    //  img[idx + k];
    //}

    int dst_width = src_width * 3 - 2;
    int dst_index = ((x * 3) + (y * dst_width * 3)) * src_channels;
    int src_index = (x + y * src_width) * src_channels;
    //int dst_height = src_height * 3 - 2;

    //long int dst_elements = dst_width * dst_height * src_channels;
    //long int src_elements = src_width * src_height * src_channels;

    int src_stride = src_width * src_channels;
    int dst_stride = dst_width * src_channels;

    

    // if invalid location do nothing.
    //if (i >= dst_width || j >= dst_height) // is that width or width-1?
    if (x >= src_width || y >= src_height)
        return;

    // all channels for a pixel are grouped together. To access an adjacent pixel, you must add by the number of channels.
    for (int k = 0; k < src_channels; k++) {

        //int dst_index = (j * 21 + i * 3) + k; // this is strictly for my predefined dst width and height (*3 -2)
        //int src_index = (j * src_width + i) + k;

        // transfer known src values to dst
        // to access different channels, the number of elements of the src/dst image must be added to the respective array index.
        dst[dst_index+k] = src[src_index+k];

        // vertical comparison acts on src image and applies values to dst image
        int y_diff = src[src_index + src_stride+k] - src[src_index+k];
        if (y_diff < threshold) { // apply third-average
           // linear fill
            int step = y_diff / 3;
            dst[dst_index + dst_stride+k] = src[src_index+k] + step;
            dst[dst_index + 2 * dst_stride+k] = src[src_index+k] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_stride+k] = src[src_index+k];
            dst[dst_index + 2 * dst_stride+k] = src[src_index + src_stride+k];
        }

        __syncthreads();

        // horizontal
        // I know this is painfully inefficient. 
        int x_diff_0 = src[src_index+k] - src[src_index + src_channels+k];
        int x_diff_1 = dst[dst_index + dst_stride + k] - dst[dst_index + dst_stride + src_channels + k];
        int x_diff_2 = dst[dst_index + 2 * dst_stride + k] - dst[dst_index + 2 * dst_stride + src_channels + k];
        int step = 0;

        if (x_diff_0 < threshold) { // apply third-average
            // linear fill
            step = x_diff_0 / 3;
            dst[dst_index + src_channels + k] = src[src_index + k] + step;
            dst[dst_index + 2*src_channels + k] = src[src_index + k] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + src_channels] = src[src_index];
            dst[dst_index + 2 * src_channels] = src[src_index + src_channels];
        }

        if (x_diff_1 < threshold) { // apply third-average
            // linear fill
            step = x_diff_1 / 3;
            dst[dst_index + dst_stride + src_channels + k] = dst[dst_index + dst_stride + k] + step;
            dst[dst_index + dst_stride + 2 * src_channels + k] = dst[dst_index + dst_stride + k] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_stride + src_channels + k] = dst[dst_index + dst_stride + k];
            dst[dst_index + dst_stride + 2 * src_channels + k] = dst[dst_index + dst_stride + 3 + k];
        }

        if (x_diff_2 < threshold) { // apply third-average
            // linear fill
            step = x_diff_2 / 3;
            dst[dst_index + 2 * dst_stride + src_channels + k] = dst[dst_index + 2 * dst_stride + k] + step;
            dst[dst_index + 2 * dst_stride + 2 * src_channels + k] = dst[dst_index + 2 * dst_stride + k] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + 2 * dst_stride + src_channels + k] = dst[dst_index + 2 * dst_stride + k];
            dst[dst_index + 2 * dst_stride + 2 * src_channels + k] = dst[dst_index + 2 * dst_stride + 3 + k];
        }
        __syncthreads();
    }
    __syncthreads();
}
