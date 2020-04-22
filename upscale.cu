// Author: Noah Van Der Weide
// 3/30/2020

// upscale the image by doubling height and width
// fill in empty areas according to neighboring pixels and difference thresholds

// THREE APPROACHES
// FIRST APPROACH:
// each thread will process one pixel
// SECOND APPROACH:
// each thread will process one original pixel and surrounding pixels
// THIRD APPROACH:
// each thread will process one original pixel and pixels to the right and below

// Two filling approaches:
// First:
// Tackle everything at once.
// Second:
// Stretch out original image and fill in adjacent pixels with original pixel value,
// Then go through and SAXPY if original pixel differences aren't too great.


// dimension of image: upper left = (0,0), bottom right = (width-1, height-1)

// *img_original is the original image
// *img_new width = *img_original width * 3 - 2
// *img_new width = *img_original height * 3 - 2


// 8 bits per color (0 - 255)
// upscale function is called independently for each color.
// this allows it to be faster for black and white images as it only needs to be called once.
// Can therefore also be applied to images which use a different color map than RGB (JPEG, for example).

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "upscale.cuh"



__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_height, int src_width, int src_channels, int threshold) {
    //__global__ void upscale(int src_width, int src_height, int threshold, unsigned int *img_original, unsigned int *img_new){

      // not using shared memory right now
      // there is 48 KB of shared memory available.
      // images are typically more than that, so I'll have to think about how it could be implemented
      //extern __shared__ unsigned char pic[];
      //unsigned char * imgData = pic;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // not relevant to code function, but shows how a thread could access a pixel in every channel.
    // pixel values are from 0 to 255.
    //for (int k = 0; k < channels; k++){
    //  img[idx + k];
    //}

    int dst_width = src_width * 3 - 2;
    int dst_height = src_height * 3 - 2;

    long int dst_elements = dst_width * dst_height * src_channels;
    long int src_elements = src_width * src_height * src_channels;

    // if invalid location do nothing.
    //if (i >= dst_width || j >= dst_height) // is that width or width-1?
    if (i >= src_width || j >= src_height)
        return;

    for (int k = 0; k < src_channels; k++) {

        int dst_index = (j * 21 + i * 3) + (k*dst_elements); // this is strictly for my predefined dst width and height (*3 -2)
        int src_index = (j * src_width + i) + (k*src_elements);

        // transfer known src values to dst
        // to access different channels, the number of elements of the src/dst image must be added to the respective array index.
        dst[dst_index] = src[src_index];

        // vertical comparison acts on src image and applies values to dst image
        int y_diff = src[src_index + src_width] - src[src_index];
        if (y_diff < threshold) { // apply third-average
           // linear fill
            int step = y_diff / 3;
            dst[dst_index + dst_width] = src[src_index] + step;
            dst[dst_index + 2 * dst_width] = src[src_index] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_width] = src[src_index];
            dst[dst_index + 2 * dst_width] = src[src_index + src_width];
        }

        __syncthreads();

        // horizontal
        // I know this is painfully inefficient. 
        int x_diff_0 = src[src_index] - src[src_index + 1];
        int x_diff_1 = dst[dst_index + dst_width] - dst[dst_index + dst_width + 1];
        int x_diff_2 = dst[dst_index + 2 * dst_width] - dst[dst_index + 2 * dst_width + 1];
        int step = 0;
        if (x_diff_0 < threshold) { // apply third-average
            // linear fill
            step = x_diff_0 / 3;
            dst[dst_index + 1] = src[src_index] + step;
            dst[dst_index + 2] = src[src_index] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + 1] = src[src_index];
            dst[dst_index + 2] = src[src_index + 1];
        }

        if (x_diff_1 < threshold) { // apply third-average
            // linear fill
            step = x_diff_1 / 3;
            dst[dst_index + dst_width + 1] = dst[dst_index + dst_width] + step;
            dst[dst_index + dst_width + 2] = dst[dst_index + dst_width] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_width + 1] = dst[dst_index + dst_width];
            dst[dst_index + dst_width + 2] = dst[dst_index + dst_width + 3];
        }

        if (x_diff_1 < threshold) { // apply third-average
            // linear fill
            step = x_diff_2 / 3;
            dst[dst_index + 2 * dst_width + 1] = dst[dst_index + 2 * dst_width] + step;
            dst[dst_index + 2 * dst_width + 1] = dst[dst_index + 2 * dst_width] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + 2 * dst_width + 1] = dst[dst_index + 2 * dst_width];
            dst[dst_index + 2 * dst_width + 1] = dst[dst_index + 2 * dst_width + 3];
        }
        __syncthreads();
    }
    __syncthreads();
}

void upscale(unsigned char* dst, unsigned char* src, int src_height, int src_width, int src_channels, int threshold){
  unsigned char * dev_src = NULL;
  unsigned char * dev_dst = NULL;

  int dst_width = src_width * 3 - 2;
  int dst_height = src_height * 3 - 2;
  int dst_elements = dst_width * dst_height * src_channels;
  int src_elements = src_width * src_height * src_channels;
  int dst_size = dst_elements * sizeof(unsigned char);
  int src_size = src_elements * sizeof(unsigned char);

  // allocate memory in GPU
  cudaMalloc((void**)&dev_dst, dst_size);
  cudaMalloc((void**)&dev_src, src_size);
  //cudaMallocManaged(&dst, dst_size);
  //cudaMallocManaged(&src, src_size);

  // copy data from CPU to GPU
  cudaMemcpy(dev_dst, dst, dst_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_src, src, src_size, cudaMemcpyHostToDevice);

  //dim3 grid_img(src_width, src_height);
  //int blocks = (src_elements * 1023) / 1024; // allocate just enough blocks to accomodate all threads. Max of 1024 threads per block.
  
  dim3 dimBlock(32, 32);
  dim3 dimGrid((dst_width + 31) / 32, (dst_height + 31) / 32);
  
  upscale_CUDA<<<dimGrid, dimBlock>>> (dev_src, src_height, src_width, src_channels, threshold); // <<<blocks, threads per block, shared mem>>>
  //upscale_CUDA <<<dimGrid, dimBlock >>> (src, src_height, src_width, src_channels, threshold); // <<<blocks, threads per block, shared mem>>>

  cudaDeviceSynchronize();

  // copy data back from GPU to CPU
  cudaMemcpy(dev_dst, dst, dst_size, cudaMemcpyDeviceToHost);

  // free GPU
  cudaFree(dev_dst);
  cudaFree(dev_src);

}
