#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}
__global__ void mandelKernel(int *deviceans,float lowerX,float lowerY,float stepX,float stepY,int resX,int resY,int maxIterations)
     {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    int localx,localy;
    localx=blockIdx.x*blockDim.x + threadIdx.x;
    localy=blockIdx.y*blockDim.y+threadIdx.y;

    float tmpx,tmpy;
    tmpx = lowerX + localx*stepX;
    tmpy = lowerY + localy*stepY;
    int ans = mandel(tmpx,tmpy,maxIterations);

    deviceans[resX* localy + localx] = ans;

}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int *deviceans;
    size_t pitch;
    int *hostans;
    dim3 threadperblock(16,16);
    dim3 numblocks(resX/16,resY/16);
    //cudaMalloc(&deviceans,resX*resY*sizeof(int));
    cudaHostAlloc(&hostans,sizeof(int)*resX*resY,cudaHostAllocMapped);
    cudaMallocPitch(&deviceans,&pitch,resX*sizeof(int),resY);
    mandelKernel<<<numblocks,threadperblock>>>(deviceans,lowerX,lowerY,stepX,stepY,resX,resY,maxIterations);
    cudaDeviceSynchronize();
    cudaMemcpy(hostans,deviceans,resY*resX*sizeof(int),cudaMemcpyDeviceToHost);
    
    for(int i=0;i<resY;++i){
      for(int j = 0;j<resX;++j){
        img[i*resX+j]=hostans[i*resX+j];
      }
    }
    cudaFree(deviceans);
    cudaFreeHost(hostans);
}
