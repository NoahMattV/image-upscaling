// Author: Noah Van Der Weide
// main.cpp for image-upscaling with CUDA

#include <iostream>
#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
// Linker > Input > Additional Dependencies > add "opencv_world430.lib" (use opencv_world430d.lib for debug)
// ok and apply

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	// load image
	string filename;
	cout << "Enter the filename of the image (e.g. peppers.png)...";
	getline(cin, filename);
	string image_path = samples::findFile(filename);
	Mat src = imread(image_path, IMREAD_COLOR);

	// check if image loaded properly
	if (src.empty()) {
		cout << "Could not read image: " << image_path << endl;
		return 1;
	}

	// get threshold
	unsigned int threshold;
	cout << "Above a specified threshold, adjacent pixels will copy nearest neighbor.\nAdjacent pixels below threshold will fill with a linear gradient.\n";
	cout << "Choose threshold (0 - 255)... ";
	cin >> threshold;

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

	cout << "Loaded " << image_path << " --  " << src_height << ", " << src_width << " -- Channels: " << channels << endl;

	upscale(src.data, dst.data, src_height, src_width, dst_height, dst_width, channels, threshold);

	// create output image.
	imshow("source", src);
	imshow("output", dst);
	imwrite("upscaled_image.png", dst);
	waitKey(0);

	return 0;
}
