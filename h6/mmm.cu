// mmm.cu

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
	int dimension1;
	int dimension2;	
} ArrayMetadata2D;

// Metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// Pointers for input and output arrays in the host memory  
float *A, *B, *C, *C_CPU;
// Pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

// TODO: tweak these?????? LOL
const int BLOCK_COUNT = 16;
const int THREADS_PER_BLOCK = 256;

//----------------------------------- Host Function Definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void computeGpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error); 
void check_error(cudaError e);

//----------------------------------- CUDA Function Definitions -----------------------------------------
// TODO: fix
__global__ void mmm_kernel(float *A, float *B, float *C, int Ax, int Ay, int Bx, int By);


//--------------------------------------------- CODE ----------------------------------------------------

int main(int argc, char **argv) {
	
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);

	allocateAndInitializeAB();

	// Matrix matrix multiplication in the CPU
	clock_t start = clock();	
	computeCpuMMM();
	clock_t end = clock();
	double elapsedCPU = (end - start) / (double) CLOCKS_PER_SEC;
	printf("Computation time in the CPU: %f seconds\n", elapsedCPU);

	//---------- MY ADDED STUFF ----------//
	// MMM on the GPU
	start = clock();
	computeGpuMMM();
	end = clock();
	double elapsedGPU = (end - start) / (double) CLOCKS_PER_SEC;
	printf("Computation time in the GPU: %f seconds\n", elapsedGPU);
	
	// Compute the speedup or slowdown
	if (elapsedGPU > elapsedCPU) {
		printf("\nCPU outperformed GPU by %.2fx\n", (float) elapsedGPU / (float) elapsedCPU);
	} else { 
		printf("\nGPU outperformed CPU by %.2fx\n", (float) elapsedCPU / (float) elapsedGPU);
	}
	
	// Check the correctness of the GPU results
	compareHostAndGpuOutput();

	return 0;
}

// Allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A = (float*) malloc(sizeofA);
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 1000) * 0.001; 
		}
	}
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 1000) * 0.001; 
		}
	}
}

// Allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// Copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// Do a straightforward matrix-matrix multiplication in the CPU notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {
	
	// Allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C = (float*) malloc(sizeofC);
	
	// Compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

// Function to determine if the GPU computation is done correctly by comparing the output from the GPU with that from the CPU
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int mismatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			mismatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (mismatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", mismatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}


//---------- MY ADDED STUFF ----------//
// TODO: MAKE THIS RIGHT 
// KERNEL: A GPU kernel that does MMM
__global__ void mmm_kernel(float *A, float *B, float *C, int Ax, int Ay, int Bx, int By) {

	// CURRENTLY THIS DOES A WEIRD VECTOR ADD IDK

	// Determine the index of the thread among all GPU threads	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//int threadCount = gridDim.x * blockDim.x; 

	if (i < Ax && j < By) {
		// Compute C[i][j]
		// Multiply A row i with B row j and add it to sum
		float sum = 0.0;
		for (int x = 0; x < Ax; x++) {
			sum += A[x+(i*Ay)] * B[(x*By)+j]; // Once B is transposed, flip it to be B[x][j]
		} 
		// Assign sum to C[i][j]
		C[j+(i*By)] = sum;
	}
}

// DO IT TO IT
// MMM on GPU
void computeGpuMMM() {

	// Transfer input to GPU
	clock_t start = clock();	
	copyMatricesToGPU();
	clock_t end = clock();
	double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	printf("GPU: Transfer to GPU: %f seconds\n", elapsed);
	
	// Execute the kernel to compute the vector sum on the GPU
	start = clock();

	// Note that we are using a one dimensional grid in this calculation as that is ideal for this
	// particular problem. For some other problem, a 2D or even a 3D grid may be appropriate. The
	// dimensionality of the grid is supposed to help you decompose the algorithmic logic inside the
	// GPU kernel. In particular, how you decide what thread should do what instruction. It does not 
	// affect the performance of the kernel.
	//add_vectors_kernel <<<BLOCK_COUNT, THREADS_PER_BLOCK>>> (A_GPU, B_GPU, C_GPU, N);
	
	// TODO: MAKE THIS KERNEL RIGHT
	// probs need to pass dimensions of A, B, and maybe C (can compute C dims)
	dim3 threadsPerBlock(BLOCK_COUNT, BLOCK_COUNT);
    dim3 numBlocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	mmm_kernel <<<numBlocks, threadsPerBlock>>> (A_GPU, B_GPU, C_GPU, A_MD.dimension1, A_MD.dimension2, B_MD.dimension1, B_MD.dimension2);
	
	// Make the CPU main thread waite for the GPU kernel call to complete
	cudaThreadSynchronize();  // This is only needed for timing and error-checking purposes
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	printf("GPU: Kernel Execution: %f seconds\n", elapsed);
	
	// Check for kernel errors
	check_error(cudaGetLastError());
	
	// Allocate CPU memory for the result
	// size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	// float *C_CPU = (float *) malloc(sizeofC);
	// if (C_CPU == NULL) die("Error allocating CPU memory");
	
	// Transfer result back to CPU
	start = clock();	
	copyResultFromGPU();
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	printf("GPU: Transfer from GPU: %f seconds\n", elapsed);
	
	// Free the GPU memory
	check_error(cudaFree(A_GPU));
	check_error(cudaFree(B_GPU));
	check_error(cudaFree(C_GPU));
}


/*
  TODO:

- find out warp size, block size should be a multiple of warp size

- memory coalescing? chunks are aligned in either 32, 64, or 128 bytes
  need to divide by size of float or something (like cache blocking)

- choose block_count and threads_per_block appropriately

- transpose B: minimizes bank conflicts and has better memory access

*/






