// Author: Noah Van Der Weide
// ImageUpscale.cpp for image-upscaling with CUDA
//
// 1. take in user specified PPM/PGM image
// 2. determine width and height
// 3. convert to vector/array
// 4. create new vector/array with larger dimensions
// 5. call upscale.cu function
// 6. export new upscaled image and time to complete
// 7. clean up (if needed)

#include "upscale.cuh"
#include "pgm.cuh"
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
// includes, project
// These were used in CUDA samples. I'm not sure if they would be useful or not.
//#include <helper_functions.h> // includes for SDK helper functions
//#include <helper_cuda.h>      // includes for cuda initialization and error checking


int main (int argc, char * argv[]){

  // CUDA timing parameters
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  // file handling
  char *image_filename = "lena.pgm";
  unsigned int width, height;
  // image array in cpu
  unsigned char* h_image = NULL;
  // load pgm
  cutLoadPGMub(image_filename, &h_image, &width, &height);


  cudaEventRecord(start);
  // upscale.cu here (this may have to be a kernel)
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << "\ntime (ms) = " << ms << std::endl;
  // clean up (if needed)

  return 0;
}
