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
float *A, *B, *C, *C_CPU, *B_TRANS;
// Pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;
// Pointers for padded A and B, and unpadded C_RES, for the GPU final result
float *A_PAD, *B_PAD, *C_RES;

// TODO: tweak these?????? LOL
const int BLOCK_COUNT = 32;
const int THREADS_PER_BLOCK = 512;
const int WARP_SIZE = 32;
int pad;

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
__global__ void mmm_kernel(float *A, float *B, float *C, int Ax, int Ay, int Bx, int By, int pad);


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
	// computeCpuMMM();
	clock_t end = clock();
	// double elapsedCPU = (end - start) / (double) CLOCKS_PER_SEC;
	// printf("Computation time in the CPU: %f seconds\n", elapsedCPU);

	//---------- MY ADDED STUFF ----------//
	// Transpose B
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B_TRANS = (float*) malloc(sizeofB);

	// print trans of B
	for (int j = 0; j < B_MD.dimension2; ++j) {
        for (int i = 0; i < B_MD.dimension1; ++i) {
            B_TRANS[i + j * B_MD.dimension1] = B[j + i * B_MD.dimension1];
            //printf("%f ", B[j + i * B_MD.dimension1]);
        }
        //printf("\n");
    }
    // we could switch B's dimensions, but since we only ever input square matrices, we don't care :)

    // Pad A and B
    // We know they're the same size so do it at the same time
    int Ax = A_MD.dimension1;
    int count = Ax;
    while (count % WARP_SIZE != 0) {
    	count++;
    }
    pad = count - Ax;

    // Allocate PAD arrays
    size_t sizeofPADs = (A_MD.dimension1 + pad) * A_MD.dimension2 * sizeof(float);
	A_PAD = (float*) malloc(sizeofPADs);
	B_PAD = (float*) malloc(sizeofPADs);

	int rowcount = 0;
	int padindex = 0;
	int p = 0;
    for (int i = 0; i < A_MD.dimension1 * A_MD.dimension2; i++) {
    	// Do padding because we're at the end of a row
    	if (rowcount == Ax) {
    		// Add however much padding there is
    		while (p < pad) {
    			A_PAD[padindex] = 0.0;
    			B_PAD[padindex] = 0.0;
    			p++;
    			padindex++;
    		}
    		p = 0;
    		rowcount = 0;
    		i--;
    	}
    	else {
    		A_PAD[padindex] = A[i];
    		B_PAD[padindex] = B_TRANS[i];
    		padindex++;
	    	rowcount++;
	    }
    }


    // print B first
	// for (int i = 0; i < B_MD.dimension1 * B_MD.dimension2; i++) {
	// 	printf("%f ", B_TRANS[i]);
	// }
	// printf("\n\n");
	// // print B_TRANS
	// for (int i = 0; i < (A_MD.dimension1 + pad) * A_MD.dimension2; i++) {
	// 	printf("%f ", B_PAD[i]);
	// }
	// printf("\n\n");

	// MMM on the GPU
	start = clock();
	computeGpuMMM();
	end = clock();
	double elapsedGPU = (end - start) / (double) CLOCKS_PER_SEC;
	printf("Computation time in the GPU: %f seconds\n", elapsedGPU);
	
	// Compute the speedup or slowdown
	// if (elapsedGPU > elapsedCPU) {
	// 	printf("\nCPU outperformed GPU by %.2fx\n", (float) elapsedGPU / (float) elapsedCPU);
	// } else { 
	// 	printf("\nGPU outperformed CPU by %.2fx\n", (float) elapsedCPU / (float) elapsedGPU);
	// }
	
	// Print out CPU result
	// for (int i = 0; i < C_MD.dimension1 * C_MD.dimension2; i++) {
	// 	printf("%f ", C[i]);
	// }
	// printf("\n\n");
	// // Print out GPU result, which is C_CPU
	// for (int i = 0; i < (C_MD.dimension1 + pad) * C_MD.dimension2; i++) {
	// 	printf("%f ", C_CPU[i]);
	// }
	// printf("\n\n");

	// Copy C_CPU to a smaller C_RES
	
	// Allocate C_RES to be normal size of C
	size_t sizeofCRES = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_RES = (float*) malloc(sizeofCRES);

	int offset = 0;
	for (int i = 0; i < C_MD.dimension1 * C_MD.dimension2; i++) {
		if (i % C_MD.dimension1 == 0 && i != 0) {
			offset += pad;
		}
		C_RES[i] = C_CPU[i + offset];
	}

	// Print C_REST
	// for (int i = 0; i < C_MD.dimension1 * C_MD.dimension2; i++) {
	// 	printf("%f ", C_RES[i]);
	// }


	// Check the correctness of the GPU results
	//compareHostAndGpuOutput();

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
	
	size_t sizeofA = (A_MD.dimension1 + pad) * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A_PAD, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = (B_MD.dimension1 + pad) * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B_PAD, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = (C_MD.dimension1 + pad) * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// Copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = (C_MD.dimension1+pad) * C_MD.dimension2 * sizeof(float);
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
		if (fabs(C[i] - C_RES[i]) > 0.01) {
			mismatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_RES[i]);
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
__global__ void mmm_kernel(float *A, float *B, float *C, int Ax, int Ay, int Bx, int By, int pad) {

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
			//sum += A[x+(i*Ay)] * B[(x*By)+j]; // for when B is NOT transposed
			sum += A[x+(i*(Ay+pad))] * B[x+(j*(By+pad))]; // for when B IS transposed
		} 
		// Assign sum to C[i][j]
		C[j+(i*(By+pad))] = sum;
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
	mmm_kernel <<<numBlocks, threadsPerBlock>>> (A_GPU, B_GPU, C_GPU, A_MD.dimension1, A_MD.dimension2, B_MD.dimension1, B_MD.dimension2, pad);
	
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

- transpose B: minimizes bank conflicts and has better memory access

- memory coalescing? chunks are aligned in either 32, 64, or 128 bytes (probs 128)
  need to divide by size of float or something (like cache blocking)
  - floats are 4 bytes
  - warp size is 32
  - all data structures need to be 128 (32*4) byte aligned (aka 32 float aligned)
  - compute padding for A and B
  	+ find next value greater than or equal to Ax that's a multiple of 32, call it next32
  	+ take next32 - Ax to get how much padding you need
  	+ pad A by copying it and adding zeroes as padding to each row
  	+ same process for padding B, and we know its dimensions are the same as A

- find out warp size, block size should be a multiple of warp size (probs 32 or something)

- choose block_count and threads_per_block appropriately

*/


