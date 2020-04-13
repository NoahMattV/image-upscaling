// Author: Noah Van Der Weide
// main.cpp for image-upscaling with CUDA
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
#include <device_launch_parameters.h> // do I need this?

// opencv is used for reading and saving images
// this will allow the use of jpegs, pngs, bmps, ppms, etc.
// see below for instructions on adding these libraries
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// may need to change configuration of project
// Active solution configuration: Release (debug is much slower)
// Active solution platform: x64

// don't forget to add opencv libraries in visual studio:
// right click main file on right side > properties > C/C++ > additional include directories > edit > find OpenCV-2.4.9 (or whatever version you have) > select 'Include' folder.
// while still in "additional include directories", hit 'new folder' and specify opencv and opencv2 subdirectories of include as well
// ...\include
// ...\include\opencv
// ...\include\opencv2
// hit 'apply'
// Properties > Linker > Additional Library Directories > Add OpenCV\x64\vc12 (or latest version)\lib
// Linker > Input > Additional Dependencies > add "opencv_core249.lib" and "opencv_highgui249.lib" (replace 249 with whatever version you have)
// ok and apply


// includes, project
// These were used in CUDA samples. I'm not sure if they would be useful or not.
//#include <helper_functions.h> // includes for SDK helper functions
//#include <helper_cuda.h>      // includes for cuda initialization and error checking


using namespace std;
using namespace cv;

int main (int argc, char * argv[]){

  unsigned int threshold = atoi(argv[1]);

  // CUDA timing parameters
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float ms;

  // file handling using opencv
  // IMREAD_COLOR loads image in BGR 8-bit format
  // IMREAD_UNCHANGED includes alpha channel
  // IMREAD_GRAYSCALE loads as intensity 0-1
  string image_path = samples::findFile("peppers.png");
  Mat img = imread(image_path, IMREAD_COLOR);

  // check if image loaded properly
  if(img.empty()){
    cout << "Could not read image: " << image_path << endl;
    return 1;
  }

  unsigned int height = img.rows;
  unsigned int width = img.cols;

  cout << "Loaded " << image_path << "--  " << height << ", " << width << " -- Channels: " << img.channels() << endl;

  // call upscale function here



  // save new image
  imwrite("upscaled_image.png", new_img);
  system("pause");

  cudaEventRecord(start);
  // upscale.cu here (this may have to be a kernel)
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

  std::cout << "\ntime (ms) = " << ms << std::endl;
  // clean up (if needed)

  return 0;
}