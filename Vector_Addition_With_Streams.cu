#include	<wb.h>

#define SegSize 1024

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index >= len) return;
	
	out[index] = in1[index] + in2[index];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
	
    float * dInputA_0;
	float * dInputB_0;
	float * dOutput_0;
	
	float * dInputA_1;
	float * dInputB_1;
	float * dOutput_1;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	
	// allocate GPU memory
	cudaMalloc((void **)&dInputA_0, SegSize*sizeof(float));
	cudaMalloc((void **)&dInputB_0, SegSize*sizeof(float));
	cudaMalloc((void **)&dOutput_0, SegSize*sizeof(float));
	
	cudaMalloc((void **)&dInputA_1, SegSize*sizeof(float));
	cudaMalloc((void **)&dInputB_1, SegSize*sizeof(float));
	cudaMalloc((void **)&dOutput_1, SegSize*sizeof(float));
	
	// create streams
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	for(int i=0;i<inputLength;i+=SegSize*2)
	{
		int left = inputLength - i;
		int s0 = 0, s1 = 0;
		
		if(left >= 2*SegSize) { s0 = s1 = SegSize; }
		else if(left >= SegSize)   { s0 = SegSize; s1 = left-SegSize; }
		else { s0 = left; s1 = 0; }
		
		cudaMemcpyAsync(dInputA_0, hostInput1 + i, s0*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dInputB_0, hostInput2 + i, s0*sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(dInputA_1, hostInput1 + i + SegSize, s1*sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dInputB_1, hostInput2 + i + SegSize, s1*sizeof(float), cudaMemcpyHostToDevice, stream1);
		
		vecAdd<<<SegSize/256, 256, 0, stream0>>>(dInputA_0, dInputB_0, dOutput_0, s0);
		vecAdd<<<SegSize/256, 256, 0, stream1>>>(dInputA_1, dInputB_1, dOutput_1, s1);
		
		cudaMemcpyAsync(hostOutput+i, dOutput_0, s0*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput+i+SegSize, dOutput_1, s1*sizeof(float), cudaMemcpyDeviceToHost, stream1);
	}		


    wbSolution(args, hostOutput, inputLength);
	
	// free GPU memory
	cudaFree(dInputA_0);
	cudaFree(dInputB_0);
	cudaFree(dOutput_0);
	
	cudaFree(dInputA_1);
	cudaFree(dInputB_1);
	cudaFree(dOutput_1);
	
	

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

