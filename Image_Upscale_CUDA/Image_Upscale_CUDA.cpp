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

// OpenCV has a resize function which performs a very similar task with adjustable dimensions. They do not, however, utilize GPU or allow for a threshold.
// It can also be very slow given the interpolation method. 


#include <iostream>
#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>

// opencv is used for reading and saving images
// this will allow the use of jpegs, pngs, bmps, ppms, etc.
// see below for instructions on adding these libraries

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include "Upscale_CUDA.h"

// may need to change configuration of project
// Active solution configuration: Release (debug is much slower)
// Active solution platform: x64

// don't forget to add opencv libraries in visual studio:
// right click main file on right side > properties > C/C++ > additional include directories > edit > find OpenCV-4.3.0 (or whatever version you have) > select 'Include' folder.
// while still in "additional include directories", hit 'new folder'
// ...build\include
// hit 'apply'
// Properties > Linker > Additional Library Directories > Add OpenCV\x64\vc15 (or latest version)\lib
// Linker > Input > Additional Dependencies > add "opencv_world430.lib" (replace 430 with whatever version you have)
// ok and apply



using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    //int t_in = atoi(argv[1]);
    //unsigned char threshold = (unsigned char)t_in;
    unsigned char threshold = 50;
    /*
    // CUDA timing parameters
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    */

    // file handling using opencv
    // IMREAD_COLOR loads image in BGR 8-bit format
    // IMREAD_UNCHANGED includes alpha channel
    // IMREAD_GRAYSCALE loads as intensity 0-1

    // load image
    string image_path = samples::findFile("peppers.png");
    Mat src = imread(image_path, IMREAD_COLOR);


    // check if image loaded properly
    if (src.empty()) {
        cout << "Could not read image: " << image_path << endl;
        return 1;
    }

    // ------------------------------------------
    // properties of the source and upscaled image
    // ------------------------------------------

    // input dimensions
    int src_height = src.rows;
    int src_width = src.cols;

    // channels (e.g. Red, Green, Blue)
    int channels = src.channels();
    //int type = src.type; // CV_8UC3?

    // output dimensions
    int dst_height = src_height * 3 - 2;
    int dst_width = src_width * 3 - 2;



    // create new image with same datatype as input
    Mat dst(dst_height, dst_width, CV_8UC3, Scalar(0, 0, 0));
    //Mat dst(dst_height, dst_width, type);

    cout << "Loaded " << image_path << " --  " << src_height << ", " << src_width << " -- Channels: " << channels << endl;

    upscale(src.data, dst.data, src_height, src_width, dst_height, dst_width, channels, threshold);


    // create output image. I might not need another Mat -- just use 'dst' instead of 'output'
    //Mat output = Mat(dst_height, dst_width, type, dst);
    imshow("source", src);
    imshow("output", dst);
    imwrite("upscaled_image.png", dst);
    waitKey(0);

    //std::cout << "\ntime (ms) = " << ms << std::endl;

    return 0;
}
