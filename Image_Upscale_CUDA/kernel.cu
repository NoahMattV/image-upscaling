// Author: Noah Van Der Weide
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Upscale_CUDA.h"

#define THREADS_PER_BLOCK 64

__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels, unsigned char threshold);

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
    int blocks = (src_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // allocate memory in GPU
    cudaMalloc((void**)&dev_dst, dst_size);
    cudaMalloc((void**)&dev_src, src_size);
    // used for shared memory if eventually implemented
    //cudaMallocManaged(&dst, dst_size);
    //cudaMallocManaged(&src, src_size);

    // copy data from CPU to GPU
    cudaMemcpy(dev_dst, dst, dst_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_src, src, src_size, cudaMemcpyHostToDevice);

    // start timer for performance evaluation
    //cudaEventRecord(start);

    // call upscale function
    //upscale_CUDA<<<blocks, THREADS_PER_BLOCK>>>  (dev_dst, dev_src, src_elements, src_width, src_height, threshold); // <<<blocks, threads per block, shared mem>>>
    dim3 grid(src_width, src_height);
    Upscale_CUDA << <grid, 1 >> > (dev_dst, dev_src, src_width, src_height, channels, threshold);
    cudaDeviceSynchronize();

    // end timer
    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    */
    // copy data back from GPU to CPU
    cudaMemcpy(dst, dev_dst, dst_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(src, dev_src, dst_size, cudaMemcpyDeviceToHost); // might not need this

    // free GPU
    cudaFree(dev_dst);
    cudaFree(dev_src);

}

__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_width, int src_height, int src_channels, unsigned char threshold) {

    // not using shared memory right now
    // there is 48 KB of shared memory available.
    // images are typically more than that, so I'll have to think about how it could be implemented
    //extern __shared__ unsigned char pic[];

  //int pixel = blockIdx.x * blockdim.x + threadIdx.x;



    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // not relevant to code function, but shows how a thread could access a pixel in every channel.
    // pixel values are from 0 to 255.
    //for (int k = 0; k < channels; k++){
    //  img[idx + k];
    //}

    int dst_width = src_width * 3 - 2;
    //int dst_height = src_height * 3 - 2;

    //long int dst_elements = dst_width * dst_height * src_channels;
    //long int src_elements = src_width * src_height * src_channels;

    int src_stride = src_width * src_channels;
    int dst_stride = dst_width * src_channels;

    // if invalid location do nothing.
    //if (i >= dst_width || j >= dst_height) // is that width or width-1?
    if (i >= src_width || j >= src_height)
        return;

    // all channels for a pixel are grouped together. To access an adjacent pixel, you must add by the number of channels.
    for (int k = 0; k < src_channels; k++) {

        int dst_index = (j * 21 + i * 3) + k; // this is strictly for my predefined dst width and height (*3 -2)
        int src_index = (j * src_width + i) + k;

        // transfer known src values to dst
        // to access different channels, the number of elements of the src/dst image must be added to the respective array index.
        dst[dst_index] = src[src_index];

        // vertical comparison acts on src image and applies values to dst image
        int y_diff = src[src_index + src_stride] - src[src_index];
        if (y_diff < threshold) { // apply third-average
           // linear fill
            int step = y_diff / 3;
            dst[dst_index + dst_stride] = src[src_index] + step;
            dst[dst_index + 2 * dst_stride] = src[src_index] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_stride] = src[src_index];
            dst[dst_index + 2 * dst_stride] = src[src_index + src_stride];
        }

        __syncthreads();

        // horizontal
        // I know this is painfully inefficient. 
        int x_diff_0 = src[src_index] - src[src_index + src_channels];
        int x_diff_1 = dst[dst_index + dst_stride] - dst[dst_index + dst_stride + src_channels];
        int x_diff_2 = dst[dst_index + 2 * dst_stride] - dst[dst_index + 2 * dst_stride + src_channels];
        int step = 0;

        if (x_diff_0 < threshold) { // apply third-average
            // linear fill
            step = x_diff_0 / 3;
            dst[dst_index + 1] = src[src_index] + step;
            dst[dst_index + 2] = src[src_index] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + src_channels] = src[src_index];
            dst[dst_index + 2 * src_channels] = src[src_index + src_channels];
        }

        if (x_diff_1 < threshold) { // apply third-average
            // linear fill
            step = x_diff_1 / 3;
            dst[dst_index + dst_stride + src_channels] = dst[dst_index + dst_stride] + step;
            dst[dst_index + dst_stride + 2 * src_channels] = dst[dst_index + dst_stride] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + dst_stride + src_channels] = dst[dst_index + dst_stride];
            dst[dst_index + dst_stride + 2 * src_channels] = dst[dst_index + dst_stride + 3];
        }

        if (x_diff_2 < threshold) { // apply third-average
            // linear fill
            step = x_diff_2 / 3;
            dst[dst_index + 2 * dst_stride + src_channels] = dst[dst_index + 2 * dst_stride] + step;
            dst[dst_index + 2 * dst_stride + 2 * src_channels] = dst[dst_index + 2 * dst_stride] + step * 2;
        }
        else { // nearest neighbor
            dst[dst_index + 2 * dst_stride + src_channels] = dst[dst_index + 2 * dst_stride];
            dst[dst_index + 2 * dst_stride + 2 * src_channels] = dst[dst_index + 2 * dst_stride + 3];
        }
        __syncthreads();
    }
    __syncthreads();
}
