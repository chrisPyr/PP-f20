#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#include "cuda.h"
__device__ float conv(int filterWidth, float *filter, int imageHeight, int imageWidth,  float *inputImage,float *outputImage ){
    int halffilterSize = filterWidth / 2;
    float sum;
    int i = blockIdx.x*blockDim.x + threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k, l;

            sum = 0;
            for (k = -halffilterSize; k <= halffilterSize; k++)
            {
                for (l = -halffilterSize; l <= halffilterSize; l++)
                {
                    if (i + k >= 0 && i + k < imageHeight &&
                        j + l >= 0 && j + l < imageWidth)
                    {
                        sum += inputImage[(i + k) * imageWidth+ l+j] *
                               filter[(k + halffilterSize) * filterWidth +
                                      l + halffilterSize];
                    }
                }
            }

    return sum;
}
__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
            outputImage[i * imageWidth + j] = conv(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage );

}
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage)
{
    int filterSize = filterWidth * filterWidth;
    int *device_filter;
    float *input, *output;
    dim3 threadperblock(16,16);
    dim3 numblocks(resX/16,resY/16);
    cudaMalloc(&device_filter,sizeof(int)*filterSize);
    cudeMalloc(&input,sizeof(float)*imageHeight*imageWidth);
    cudaMalloc(&output,sizeof(float)*imageHeight*imageWidth);
    convolution<<<numblocks,threadperblock>>>(filterWidth,device_filter,imageHeight,imageWidth,input,output);
    int work_size = imageHeight*imageWidth;
    cudaMemcpy(outputImage,output,work_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(device_filter);
    cudaFree(input);
    cudaFree(output);
    free(outputImage);
}

