// Author: Noah Van Der Weide
// main.cu for image-upscaling with CUDA
//
// 1. take in user specified PPM image
// 2. determine width and height
// 3. convert to vector/array
// 4. create new vector/array with larger dimensions
// 5. call upscale.cu function
// 6. export new upscaled image and time to complete
// 7. clean up (if needed)

#include "upscale.cuh"
#include <iostream>


int main (int argc, char * argv[]){

  // CUDA timing parameters
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;




  cudaEventRecord(start);
  // upscale.cu here
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << "\ntime (ms) = " << ms << std::endl;
  // clean up (if needed)

  return 0;
}
