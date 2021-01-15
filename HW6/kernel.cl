__kernel void convolution(int filterWidth, __global const float *filter, int imageHeight, int imageWidth, __global const float *inputImage,__global float *outputImage ) 
{
    int halffilterSize = filterWidth / 2;
    float sum;
    int i = get_global_id(0)/imageWidth,j=get_global_id(0)%imageWidth, k, l;

    
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
            outputImage[i * imageWidth + j] = sum;

}
