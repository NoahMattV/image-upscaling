// pgm input/output

#ifndef PGM_CUH
#define PGM_CUH

void loadImage(int *image, int width, int height, int grayscale, string filename);
void saveImage(int *image, const int width, const int height, const int grayscale, string filename);

#endif
