#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    cl_int err;
    int filterSize = filterWidth * filterWidth;
    cl_command_queue queue = clCreateCommandQueue(*context, device[0], 0, 0);
    cl_mem device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*filterSize, filter, &err);
    cl_mem output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float)*imageHeight*imageWidth, NULL, &err);
    cl_mem input = clCreateBuffer(*context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*imageHeight*imageWidth, inputImage, NULL);
    cl_kernel kernel = clCreateKernel(*program, "convolution", 0);
   clSetKernelArg(kernel, 0 ,sizeof(int),&filterWidth );
    clSetKernelArg(kernel, 1, sizeof(cl_mem),&device_filter);
    clSetKernelArg(kernel, 2, sizeof(int),&imageHeight);
    clSetKernelArg(kernel, 3, sizeof(int),&imageWidth);
   clSetKernelArg(kernel, 4, sizeof(cl_mem),&input);
    clSetKernelArg(kernel, 5, sizeof(cl_mem),&output);
    int work_size = imageHeight*imageWidth;
    size_t local_size = 256;
    size_t global_size = ceil(work_size/(float)local_size)*local_size;
   clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,&local_size, 0, NULL, NULL);
   if (err){
        printf("%d\n",err);
   }
        clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * imageHeight*imageWidth, outputImage, 0, 0, 0);
        clReleaseMemObject(input);
    clReleaseMemObject(output);
//clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
 //   clReleaseContext(context);
}
