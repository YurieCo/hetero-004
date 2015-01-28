// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
__constant__ float deviceCDF[HISTOGRAM_LENGTH];



//@@ insert code here

float clamp(float x, float start, float end)
{
    if(x>end) return end;
    if(x<start) return start;
    return x;
}

__device__ float device_clamp(float x, float start, float end)
{
    if(x>end) return end;
    if(x<start) return start;
    return x;
}

float correct(float val, float* CDF)
{
    float y = (CDF[(uint8_t)(val*255)] - CDF[0]) / (1.0f - CDF[0]);
    return clamp(y, 0.0f, 1.0f);
}

__global__ void kernal_histogram(uint8_t *image, uint32_t *hist, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len)
    {
        unsigned int* address = &hist[image[idx]];
        atomicAdd(address, 1);
    }
}

__global__ void kernal_correct(float *inputImage, float *outputImage, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len)
    {
        float val = inputImage[idx];
        float y = (deviceCDF[(uint8_t)(val*255)] - deviceCDF[0]) / (1.0 - deviceCDF[0]);
        outputImage[idx] = device_clamp(y, 0.0f, 1.0f);
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
    float * deviceInputImageData;
    float * deviceOutputImageData;
    uint8_t *hostGreyScaleImageData;
    uint8_t *deviceGreyScaleImageData;
    uint32_t  *hostHistogram;
    uint32_t  *deviceHistogram;
    float hostCDF[HISTOGRAM_LENGTH] = {0};    

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbLog(TRACE, "The image length is ", imageWidth, " * ", imageHeight);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here

    int imageLength = imageWidth * imageHeight * imageChannels * sizeof(float);
    int grayImageLength = imageWidth * imageHeight * sizeof(uint8_t);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    hostGreyScaleImageData = (uint8_t*)malloc(grayImageLength);

    wbTime_start(Generic, "Convert to gray scale on CPU");
    for(int i=0;i<imageWidth*imageHeight;++i)
    {
        float r = hostInputImageData[i*3+0];
        float g = hostInputImageData[i*3+1];
        float b = hostInputImageData[i*3+2];
        uint8_t y = 255.0*(0.21*r + 0.71*g + 0.07*b);
        hostGreyScaleImageData[i] = y;
    }
    wbTime_stop(Generic, "Convert to gray scale on CPU");

    wbTime_start(GPU, "Copy gray image to GPU");
    cudaMalloc((void**)&deviceGreyScaleImageData, grayImageLength);
    cudaMemcpy(
        deviceGreyScaleImageData,
        hostGreyScaleImageData,         
        imageWidth*imageHeight*sizeof(uint8_t), 
        cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copy gray image to GPU");

    wbTime_start(GPU, "Compute histogram GPU");
    cudaMalloc((void**)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(uint32_t));
    cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(uint32_t));
    dim3 block(BLOCK_SIZE);
    dim3 grid((imageWidth*imageHeight + BLOCK_SIZE - 1)/BLOCK_SIZE);
    kernal_histogram<<<grid,block>>>(
        deviceGreyScaleImageData, 
        deviceHistogram, 
        imageWidth*imageHeight);
    wbTime_stop(GPU, "Compute histogram GPU");

    wbTime_start(GPU, "Copy histogram back");
    hostHistogram = (uint32_t*)malloc(HISTOGRAM_LENGTH * sizeof(uint32_t));
    cudaMemcpy(
        hostHistogram, 
        deviceHistogram, 
        HISTOGRAM_LENGTH * sizeof(uint32_t), 
        cudaMemcpyDeviceToHost);
    wbTime_stop(GPU, "Copy histogram back");
    
    wbTime_start(Generic, "Compute CDF on CPU");
    hostCDF[0] = 1.0f*hostHistogram[0]/imageWidth/imageHeight;
    for(int i=1;i<HISTOGRAM_LENGTH;++i)
        hostCDF[i] = hostCDF[i-1] + 1.0f*hostHistogram[i]/imageWidth/imageHeight;
    cudaMemcpyToSymbol(deviceCDF, hostCDF, sizeof(uint32_t)*HISTOGRAM_LENGTH);
    wbTime_stop(Generic, "Compute CDF on CPU");


    wbTime_start(Generic, "Copy Image to GPU");
    cudaMalloc((void**)&deviceInputImageData, imageLength);
    cudaMalloc((void**)&deviceOutputImageData, imageLength);    
    cudaMemcpy(
        deviceInputImageData, 
        hostInputImageData, 
        imageLength,
        cudaMemcpyHostToDevice);
    wbTime_stop(Generic, "Copy Image to GPU");

    wbTime_start(Generic, "Correct Image on GPU");
    block = dim3(BLOCK_SIZE);
    grid = dim3((imageWidth*imageHeight*imageChannels + BLOCK_SIZE - 1)/BLOCK_SIZE);
    kernal_correct<<<grid, block>>>(
        deviceInputImageData, 
        deviceOutputImageData,
        imageWidth*imageHeight*imageChannels);
    wbTime_stop(Generic, "Correct Image on GPU");


    wbTime_start(Generic, "Copy Correct Image to host");    
    cudaMemcpy(
        hostOutputImageData, 
        deviceOutputImageData,         
        imageLength,
        cudaMemcpyDeviceToHost);
    wbTime_stop(Generic, "Copy correct Image to host");

    wbSolution(args, outputImage);

    //@@ insert code here
    free(hostHistogram);
    free(hostGreyScaleImageData);
    cudaFree(deviceGreyScaleImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceHistogram);

    return 0;
}

