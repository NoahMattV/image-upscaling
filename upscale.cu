
// upscale the image by doubling height and width
// fill in empty areas according to neighboring pixels and difference thresholds

// each thread will process one pixel
// dimension of image: upper left = (0,0), bottom right = (width-1, height-1)

// *bmpOriginal is the original image
// *bmpNew width = *bmpOriginal width * 3 - 2
// *bmpNew width = *bmpOriginal height * 3 - 2


// 8 bits per color (0 - 255)
// upscale function is called independently for each color.
// this allows it to be faster for black and white images as it only needs to be called once.
// Can therefore also be applied to images which use a different color map than RGB (JPEG, for example).

int difference(int Ax, int Ay, int Bx, int By, int stride, unsigned int *bmpOriginal);


__global__ void upscale(int originalWidth, int originalHeight, int threshold, unsigned int *bmpOriginal, unsigned int *bmpNew){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

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
        bmpNew[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // just below original pixel (down + 1)
        // check difference between lower original pixel and original pixel
        // apply based on threshold
        bmpNew[j*newStride + i] = fill(i, j, i, j, i + 2, j + 2, originalWidth, threshold, bmpOriginal);
      }
      else{ // yPos == 2 two spaces below original pixel (down + 2)

      }
      break;
    case 1:
      if (yPos == 0){ // to the right of original pixel
        bmpNew[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // diagonally down and right of original pixel (right + 1, down + 1)

      }
      else{ // yPos == 2 (right + 1, down + 2)

      }
      break;
    case 2:
      if (yPos == 0){ // (right + 2)
        bmpNew[j*newStride + i] = bmpOld[j*originalStride + i];
      }
      else if (yPos == 1){ // (right + 2, down + 1)

      }
      else{ // yPos == 2 (right + 2, down + 2)

      }
      break;
    default:
      // something went wrong
      break;
  } // end switch

}

int difference(int i, int j, int Ax, int Ay, int Bx, int By, int stride, unsigned int *bmpOriginal){
  return (bmpOriginal[Ay*stride + Ax] - bmpOriginal[By*stride + Bx]);
}

int fill(int Ax, int Ay, int Bx, int By, int stride, int threshold, unsigned int *bmpOriginal){
  int diff = (bmpOriginal[Ay*stride + Ax] - bmpOriginal[By*stride + Bx]);
  int dist = 3;

  //if ((By - Ay) > (Bx - Ax))
  //  dist = (By - Ay);
  //else
  //  dist = (Bx - Ax);

  if (diff < threshold){ // apply third-average
    int step = diff/dist;
    if (((By - j) > 1) || ((Bx - i) > 1))
      return ()
  }
  else{
    return bmpOriginal[Ay*stride + Ax];
  }
}