#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILE_SIZE 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, 
									 int m,
                                     int n,
                                     int k) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
	
	__shared__ float ds_A[TILE_SIZE][TILE_SIZE];
	__shared__ float ds_B[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int COL = bx * blockDim.x + tx;
	int ROW = by * blockDim.y + ty;	
	
	float sum = 0.0;
	
	// ( m * n) X (n * k) = (m * k)
	
	int iters = (n + TILE_SIZE-1)/TILE_SIZE;
	

	for(int i = 0; i < iters; ++i)
	{
		// load to shared memory	
	
		int ax = i * TILE_SIZE + tx;
		int by = i * TILE_SIZE + ty;
		
		if (ROW < m && ax < n)			
			ds_A[ty][tx] = A[ ROW * n + ax ];
		else
			ds_A[ty][tx] = 0.0;
			
		if (COL < k && by < n )
			ds_B[ty][tx] = B[ by  * k + COL];
		else
			ds_B[ty][tx] = 0.0;
		
		__syncthreads();
	
		
		for(int k=0;k<TILE_SIZE;++k)
			sum += ds_A[ty][k] * ds_B[k][tx];
	
		__syncthreads();
	}
	
	if(COL < k && ROW < m)
		C[ROW*k + COL] = sum;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = 
	  ( float * )malloc(numCRows*numCColumns*sizeof(float));
	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
	
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
	dim3 grid((numCColumns+TILE_SIZE-1)/TILE_SIZE, (numCRows+TILE_SIZE-1)/TILE_SIZE);
	dim3 block(TILE_SIZE,TILE_SIZE);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<grid, block>>>(
		deviceA,
		deviceB,
		deviceC,
		numARows, 
		numAColumns,
		numBColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
	wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
