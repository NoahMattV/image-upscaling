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
#include "upscale.cuh"
#include <iostream>


__global__ void upscale_CUDA(unsigned char* dst, unsigned char* src, int src_height, int src_width, int src_channels, int threshold){
//__global__ void upscale(int src_width, int src_height, int threshold, unsigned int *img_original, unsigned int *img_new){

  // not using shared memory right now
  // there is 48 KB of shared memory available.
  // images are typically more than that, so I'll have to think about how it could be implemented
  //extern __shared__ unsigned char pic[];
  //unsigned char * imgData = pic;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x;
  int y = blockIdx.y;
  //int src_idx = (x + y * gridDim.x) * src_channels;


  // not relevant to code function, but shows how a thread could access a pixel in every channel.
  // pixel values are from 0 to 255.
  //for (int k = 0; k < channels; k++){
  //  img[idx + k];
  //}

  int dst_stride = 3; // this may just be dst_width or dst_height
  int src_stride = 1;

  int dst_width = src_width * 3 - 2;
  int dst_height = src_height * 3 - 2;

  // if invalid location do nothing.
  //if (i >= dst_width || j >= dst_height) // is that width or width-1?
  if (i >= src_width || j >= src_height)
      return;

  int xPos = i % 3;
  int yPos = j % 3;

  // If pixel corresponds to orignal image (%3 == 0), then just place pixel
  // Otherwise, if pixel is between two original pixels, take difference
  // If difference is less than threshold, apply third-average to pixel based on nearest pixel.
  // If difference is greater than or equal to threshold, copy nearest original pixel
  int dst_index = j * dst_stride + i;
  int src_index = j * src_stride + i;
  //
  int x_diff = src[src_index + 1] - src[src_index];
  int y_diff = src[src_index + src_stride] - src[src_index];
  int d_diff = src[src_index + src_stride + 1] - src[src_index];


  // horizontal
  if (x_diff < threshold) { // apply third-average
      // linear fill
      int step = x_diff / 3;
      dst[dst_index + 1] = src[src_index] + step;
      dst[dst_index + 2] = src[src_index] + step * 2;
  }
  else { // nearest neighbor
      dst[dst_index + 1] = src[src_index];
      dst[dst_index + 2] = src[src_index + 1];
  }

  // vertical
  if (y_diff < threshold) { // apply third-average
      // linear fill
      int step = x_diff / 3;
      dst[dst_index + 1] = src[src_index] + step;
      dst[dst_index + 2] = src[src_index] + step * 2;
  }
  else { // nearest neighbor
      dst[dst_index + 1] = src[src_index];
      dst[dst_index + 2] = src[src_index + 1];
  }

  // diagonal
  if (d_diff < threshold) { // apply third-average
      // linear fill
      int step = x_diff / 3;
      dst[dst_index + 1] = src[src_index] + step;
      dst[dst_index + 2] = src[src_index] + step * 2;
  }
  else { // nearest neighbor
      dst[dst_index + 1] = src[src_index];
      dst[dst_index + 2] = src[src_index + 1];
  }

  // off-diagonals ("down 1, over 2" and "down 2, over 1")
  if (x_diff < threshold) { // apply third-average
      // linear fill
      int step = x_diff / 3;
      dst[dst_index + 1] = src[src_index] + step;
      dst[dst_index + 2] = src[src_index] + step * 2;
  }
  else { // nearest neighbor
      dst[dst_index + 1] = src[src_index];
      dst[dst_index + 2] = src[src_index + 1];
  }
  
  /*
  switch (xPos){
    case 0:
      if (yPos == 0){ // corresponds to original image pixel
        dst[dst_index] = src[src_index];
      }
      else if (yPos == 1){ // just below original pixel (down + 1)
        // check difference between lower original pixel and original pixel
        // apply based on threshold
        // 
        
        fill(dst[dst_index], src[src_index], Ax, Ay, Bx, By, threshold);
        fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, true);
      }
      else{ // yPos == 2 two spaces below original pixel (down + 2)
        img[dst_index] = fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, false);
      }
      break;
    case 1:
      if (yPos == 0){ // to the right of original pixel
        img[dst_index] = img[src_index];
      }
      else if (yPos == 1){ // diagonally down and right of original pixel (right + 1, down + 1)
        img[j*dst_stride + i] = fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, true);
      }
      else{ // yPos == 2 (right + 1, down + 2)
        img[j*dst_stride + i] = fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, false);
      }
      break;
    case 2:
      if (yPos == 0){ // (right + 2)
        img[dst_index] = img[src_index];
      }
      else if (yPos == 1){ // (right + 2, down + 1)
        img[j*dst_stride + i] = fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, false);
      }
      else{ // yPos == 2 (right + 2, down + 2)
        img[j*dst_stride + i] = fill(i, j, i, j, i + 2, j + 2, src_width, threshold, img, false);
      }
      break;
    default:
      // something went wrong
      break;
  } // end switch
  */
}

// Ax, Ay are the coordinates for the nearest original pixel
// Bx, By are the coordinates for the second nearest original pixel
//__global__ void difference(int result, int Ax, int Ay, int Bx, int By, int stride, unsigned char *img){
//  result = (img[Ay*stride + Ax] - img[By*stride + Bx]);
//  return;
//}

// Ax, Ay are the coordinates for the nearest original pixel
// Bx, By are the coordinates for the second nearest original pixel
// i, j are the coordinates for the current pixel

// can possibly implement a SAXPY operation to fill the missing values more efficiently
// A = By-Ay or Bx-Ax or (By+Ay)-(Bx+Ax) divided by distance (in our case, 3);
// Y = Ay or Ax
// X is 1 or 2, depending on whether we have an adjacent pixel or not.

//__global__ void fill(int i, int j, int Ax, int Ay, int Bx, int By, int stride, int threshold, unsigned char *img, bool adjacent)
__global__ void fill(unsigned char dst_idx, unsigned char src_idx, int Ax, int Ay, int Bx, int By, int threshold){
  int diff = (img[Ay*stride + Ax] - img[By*stride + Bx]);
  int dist = 3;

  //if ((By - Ay) > (Bx - Ax))
  //  dist = (By - Ay);
  //else
  //  dist = (Bx - Ax);

  if (diff < threshold){ // apply third-average
    int step = diff/dist;
    if (adjacent == false) // if non-adjacent to (Ax, Ay) -- apply Ax,Ay + step*2
      return (img[Ay*stride + Ax] + step*2);
    else // adjacent pixel
      return (img[Ay*stride + Ax] + step);
  }
  else{ // threshold exceeded. Apply same value as nearest original neighbor.
    if (adjacent == false) // non-adjacent to (Ax, Ay)
      return img[By*stride + Bx];
    else // adjacent pixel
      return img[Ay*stride + Ax];
  }
}

void upscale(unsigned char* dst, unsigned char* src, int src_height, int src_width, int src_channels, int threshold){
  unsigned char * dev_src = NULL;
  unsigned char * dev_dst = NULL;

  int dst_width = src_width * 3 - 2;
  int dst_height = src_height * 3 - 2;
  int dst_size = dst_width * dst_height * src_channels;
  int src_size = src_width * src_height * src_channels;

  // allocate memory in GPU
  cudaMalloc((void**)&dev_dst, dst_size);
  cudaMalloc((void**)&dev_src, src_size);

  // copy data from CPU to GPU
  cudaMemcpy(dev_dst, dst, dst_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_src, src, src_size, cudaMemcpyHostToDevice);

  dim3 grid_img(src_width, src_height);
  upscale_CUDA<<<grid_img, 1>>>(dev_src, src_height, src_width, src_channels, threshold);

  // copy data back from GPU to CPU
  cudaMemcpy(dev_dst, dst, dst_size, cudaMemcpyDeviceToHost);

  // free GPU
  cudaFree(dev_src);
  cudaFree(dev_dst);

}
