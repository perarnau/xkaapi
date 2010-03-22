
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main(int argc, char** argv)
{

  int deviceCount; 
  int device; 
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		printf("cudaGetDeviceCount failed! CUDA Driver and Runtime version may be mismatched.\n");
		printf("\nTest FAILED!\n");
		exit(1);
	}
  for (int device = 0; device < deviceCount; ++device) 
  { 
    CUdevice cuDevice; 
    cuDeviceGet(&cuDevice, device); 
    int major, minor; 
    cuDeviceComputeCapability(&major, &minor, cuDevice); 
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
        printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

		printf("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
    #if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
    #endif
        printf("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 2020
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif


  } 
  return 0;
}
