#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_SIZE 12
#define I_TILE_SIZE 16

//@@ INSERT CODE HERE

__device__ float clamp(float v, float low, float high)
{
	if(v<low) return low;
	if(v>high) return high;
	return v;
}

float clampCPU(float v, float low, float high)
{
	if(v<low) return low;
	if(v>high) return high;
	return v;
}

__global__ void convolution2D(float *I, float *O, const float *M, int w, int h, int channels)
{
	__shared__ float S[3][I_TILE_SIZE][I_TILE_SIZE];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int O_COL = bx*O_TILE_SIZE + tx;
	int O_ROW = by*O_TILE_SIZE + ty;
	int O_index = (O_ROW*w + O_COL)*channels + bz;
	
	int I_COL = O_COL - Mask_radius;
	int I_ROW = O_ROW - Mask_radius;
	int I_index = (I_ROW*w + I_COL)*channels + bz;	
	
	
	//load to shared memory
	if(I_COL>=0 && I_ROW>=0 && I_COL < w && I_ROW < h)
		S[bz][ty][tx] = I[I_index];
	else
		S[bz][ty][tx] = 0.0f;
	
	__syncthreads();

	// do the convolution	
	float sum = 0.0f;
	
	if(tx < O_TILE_SIZE && ty < O_TILE_SIZE && O_COL < w && O_ROW < h)
	{
		for(int i=0;i<Mask_width;i++)
			for(int j=0;j<Mask_width;j++)
				sum += S[bz][ty+i][tx+j] * M[i * Mask_width + j];				
		O[O_index] = clamp(sum,0,1);
	}

	
	__syncthreads();
}

void convolution2DCPU(float *I, float *O, const float *M, int w, int h, int channels)
{
	for(int c=0;c<channels;c++)
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
			{
				float sum = 0.0;
				for(int p=-Mask_radius;p<=Mask_radius;++p)
					for(int q=-Mask_radius;q<=Mask_radius;++q)
				{
					int sy = i + p;
					int sx = j + q;
					if(sx<0 || sy<0 || sx>=w || sy>=h) continue;
					sum += I[(sy*w+sx)*channels+c] * M[(p+Mask_radius)*Mask_width + (q+Mask_radius)];
				}
				O[(i*w+j)*channels+c] = clampCPU(sum,0.0f,1.0f);
			}
}	


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	
	wbLog(TRACE, "The dimensions of the image are ", imageWidth, " x ", imageHeight, " x ", imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

	
	wbTime_start(GPU, "Doing the computation on the CPU");
	
	convolution2DCPU(hostInputImageData, hostOutputImageData, hostMaskData, imageWidth, imageHeight, imageChannels);
	
	wbTime_stop(GPU, "Doing the computation on the CPU");
	
	
	
    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 block(I_TILE_SIZE,I_TILE_SIZE);
	dim3 grid((imageWidth + O_TILE_SIZE-1) / O_TILE_SIZE, (imageHeight + O_TILE_SIZE - 1) / O_TILE_SIZE, imageChannels);
	
	convolution2D<<<grid, block>>>(deviceInputImageData, 
								   deviceOutputImageData, 
								   deviceMaskData,
								   imageWidth,
								   imageHeight,
								   imageChannels);
	
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
