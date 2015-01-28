// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
typedef unsigned char BYTE;

//@@ insert code here

float clamp(float x, float start, float end)
{
    if(x>end) return end;
    if(x<start) return start;
    return x;
}

float correct(float val, float* CDF)
{
    float y = (CDF[(BYTE)(val*255)] - CDF[0]) / (1.0f - CDF[0]);
    return clamp(y, 0.0f, 1.0f);
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
    BYTE *hostGrayScaleImageData;
    BYTE *deviceGrayScaleImageData;
    int  hostHistogram[HISTOGRAM_LENGTH] = {0};
    float hostCDF[HISTOGRAM_LENGTH] = {0};
    int  deviceHistogram[HISTOGRAM_LENGTH];

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

	wbTime_start(Generic, "Compute histogram on CPU");
    for(int i=0;i<imageWidth*imageHeight;++i)
    {
        float r = hostInputImageData[i*3+0];
        float g = hostInputImageData[i*3+1];
        float b = hostInputImageData[i*3+2];
        BYTE y = 255.0*(0.21*r + 0.71*g + 0.07*b);
        ++hostHistogram[y];
    }
	wbTime_stop(Generic, "Compute histogram on CPU");
	
	
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
