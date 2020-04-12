// Author: Noah Van Der Weide
// 3/30/2020

// PPM or PGM image format to start.

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


__global__ void upscale(unsigned char * img, int originalHeight, int originalWidth, int channels){
//__global__ void upscale(int originalWidth, int originalHeight, int threshold, unsigned int *img_original, unsigned int *img_new){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x;
  int y = blockIdx.y;
  int idx = (x + y * gridDim.x) * channels;


  // not relevant to code function, but shows how a thread could access a pixel in every channel.
  // pixel values are from 0 to 255.
  for (int k = 0; k < channels; k++){
    img[idx + k];
  }

  int newStride = 3; // this may just be newWidth or newHeight
  int originalStride = 1;

  int newWidth = originalWidth * 3 - 2;
  int newHeight = orignalHeight * 3 - 2;

  // if invalid location do nothing.
  if (i >= newWidth || j >= newHeight)
    return;

  int xPos = i % 3;
  int yPos = j % 3;

  // If pixel corresponds to orignal image (%3 == 0), then just place pixel
  // Otherwise, if pixel is between two original pixels, take difference
  // If difference is less than threshold, apply third-average to pixel based on nearest pixel.
  // If difference is greater than or equal to threshold, copy nearest original pixel

  //
  switch (xPos){
    case 0:
      if (yPos == 0){ // corresponds to original image pixel
        img_new[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // just below original pixel (down + 1)
        // check difference between lower original pixel and original pixel
        // apply based on threshold
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, true);
      }
      else{ // yPos == 2 two spaces below original pixel (down + 2)
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, false);
      }
      break;
    case 1:
      if (yPos == 0){ // to the right of original pixel
        img_new[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // diagonally down and right of original pixel (right + 1, down + 1)
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, true);
      }
      else{ // yPos == 2 (right + 1, down + 2)
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, false);
      }
      break;
    case 2:
      if (yPos == 0){ // (right + 2)
        img_new[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // (right + 2, down + 1)
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, false);
      }
      else{ // yPos == 2 (right + 2, down + 2)
        img_new[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, img_original, false);
      }
      break;
    default:
      // something went wrong
      break;
  } // end switch

}

// Ax, Ay are the coordinates for the nearest original pixel
// Bx, By are the coordinates for the second nearest original pixel
int difference(int Ax, int Ay, int Bx, int By, int stride, unsigned int *img_original){
  return (img_original[Ay*stride + Ax] - img_original[By*stride + Bx]);
}

// Ax, Ay are the coordinates for the nearest original pixel
// Bx, By are the coordinates for the second nearest original pixel
// i, j are the coordinates for the current pixel

// can possibly implement a SAXPY operation to fill the missing values more efficiently
// A = By-Ay or Bx-Ax or (By+Ay)-(Bx+Ax) divided by distance (in our case, 3);
// Y = Ay or Ax
// X is 1 or 2, depending on whether we have an adjacent pixel or not.

int fill(int i, int j, int Ax, int Ay, int Bx, int By, int stride, int threshold, unsigned int *img_original, bool adjacent){
  int diff = (img_original[Ay*stride + Ax] - img_original[By*stride + Bx]);
  int dist = 3;

  //if ((By - Ay) > (Bx - Ax))
  //  dist = (By - Ay);
  //else
  //  dist = (Bx - Ax);

  if (diff < threshold){ // apply third-average
    int step = diff/dist;
    if (adjacent == false) // if non-adjacent to (Ax, Ay) -- apply Ax,Ay + step*2
      return (img_original[Ay*stride + Ax] + step*2);
    else // adjacent pixel
      return (img_original[Ay*stride + Ax] + step);
  }
  else{ // threshold exceeded. Apply same value as nearest original neighbor.
    if (adjacent == false) // non-adjacent to (Ax, Ay)
      return img_original[By*stride + Bx];
    else // adjacent pixel
      return img_original[Ay*stride + Ax];
  }
}

void upscale_CUDA(unsigned char * input_img, int height, int width, int channels){
  unsigned char * Dev_Input_Img = NULL;

  int newWidth = width * 3 - 2;
  int newHeight = height * 3 - 2;
  int size = newWidth * newHeight * channels;
  // allocate memory in GPU
  cudaMalloc((void**)&Dev_Input_Img, size);

  // copy data from CPU to GPU
  cudaMemcpy(Dev_Input_Img, input_img, size, cudaMemcpyHostToDevice);

  dim3 grid_img(width, height);
  upscale<<<grid_img, 1>>>(Dev_Input_Img, height, width, channels);

  // copy data back from GPU to CPU
  cudaMemcpy(Dev_Input_Img, input_img, size, cudaMemcpyDeviceToHost);

  // free GPU
  cudaFree(Dev_Input_Img);

}
