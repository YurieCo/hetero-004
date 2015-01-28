// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//@@ insert code here

float clamp(float x, float start, float end)
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
    uint8_t *hostGrayScaleImageData;
    uint8_t *deviceGrayScaleImageData;
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
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    hostGrayScaleImageData = (uint8_t*)malloc(imageWidth*imageHeight*sizeof(uint8_t));

    wbTime_start(Generic, "Convert to gray scale on CPU");
    for(int i=0;i<imageWidth*imageHeight;++i)
    {
        float r = hostInputImageData[i*3+0];
        float g = hostInputImageData[i*3+1];
        float b = hostInputImageData[i*3+2];
        uint8_t y = 255.0*(0.21*r + 0.71*g + 0.07*b);
        hostGrayScaleImageData[i] = y;
    }
    wbTime_stop(Generic, "Convert to gray scale on CPU");

    wbTime_start(GPU, "Copy gray image to GPU");
    cudaMalloc((void**)&deviceGrayScaleImageData, imageWidth*imageHeight*sizeof(uint8_t));
    cudaMemcpy(
        deviceGrayScaleImageData,
        hostGrayScaleImageData,         
        imageWidth*imageHeight*sizeof(uint8_t), 
        cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copy gray image to GPU");

    wbTime_start(GPU, "Compute histogram GPU");
    cudaMalloc((void**)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(uint32_t));
    cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(uint32_t));
    dim3 block(BLOCK_SIZE);
    dim3 grid((imageWidth*imageHeight + BLOCK_SIZE - 1)/BLOCK_SIZE);
    kernal_histogram<<<grid,block>>>(
        deviceGrayScaleImageData, 
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
    wbTime_stop(Generic, "Compute CDF on CPU");

    wbTime_start(Generic, "Correct Image on CPU");
    for(int i=0;i<imageWidth*imageHeight*imageChannels;++i)
    {
        hostOutputImageData[i] = correct(hostInputImageData[i], hostCDF);
    }
    wbTime_stop(Generic, "Correct Image on CPU");

    wbSolution(args, outputImage);

    //@@ insert code here
    free(hostHistogram);

    return 0;
}

