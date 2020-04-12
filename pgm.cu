#include "pgm.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void loadImage(int *image, int width, int height, int grayscale, string filename){
  string line;
  string dim[2];

  cout << "\n Loading Image... " << &filename << endl;

  ifstream image_in(&filename);
  if (image_in){
    getline(image_in, line); // the magic number for PGM/PPM
    if (line == "P2"){ // gray-tone image in ASCII format
      cout << "P2 ASCII PPM File Found! (.pgm)" << endl;
    }
    else if (line == "P3"){ // color image in ASCII format (PGM)
      cout << "P3 ASCII PGM File Found! (.pgm)" << endl;
    }
    else{
      cout << "File is not in correct format. Need a P2 ASCII PPM or P3 ASCII PGM (.pgm)" << endl;
      return;
    }

    // skip comments (marked with a '#')
    getline(image_in, line);
    while (line.at(0) == '#'){
        getline(image_in, line);
    }

    // get dimensions of image
    stringstream ssin(line); // break line into a string array
    while (ssin && i < 2){
      ssin >> dim[i]; // convert to ints
      i++;
    }

    // convert strings to ints
    &width = stoi(dim[0]);
    &height = stoi(dim[0]);

    cout << "Image is " << width << " x " << height << endl;

    // grayscale values
    getline(image_in, line);
    &grayscale = stoi(line);
    cout << "The grayscale range is 0 to " << grayscale << endl;

    // store into 2D int array
    for (int i = 0; i < width * height; i++){
      image_in >> ws; // extracts whitespace characters. Do I need this?
      getline(image_in, line, ' ');
      image[i] = stoi(line);
    }
    image_in.close(); // close ifstream

    cout << "\nImage Loaded!" << endl;
  } // if (image_in)
}

void saveImage(int *image, const int width, const int height, const int grayscale, string filename){
  cout << "Saving image as " << filename << ".pgm to " << endl;
  fstream image_out;
  image_out.open("Output/" + filename + ".pgm", fstream::out);

  image_out << "P2" << endl;
  image_out << "#P2/ASCII PGM (Portable Gray Map) Filter Output" << endl;

  image_out << to_string(width) + " " + to_string(height) << endl;
	image_out << to_string(grayscale) << endl;

  int total = width * height;

  for (int i = 0; i < total; i++){
    image_out << to_string(image[i]) + " ";
  }

  image_out.close();

  cout << "\nImage Saved!\n" << endl;
}
