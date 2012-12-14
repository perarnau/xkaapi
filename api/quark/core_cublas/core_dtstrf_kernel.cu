
#include <cuda.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  64
#endif

__global__ void core_dtstrf_cmp_kernel(double* A, double* U, double* dev_ptr)
{
  if(fabs(A[0]) > fabs(U[0]))
  {
    dev_ptr[0] = 1.0f;
  }
  else
  {
    dev_ptr[0] = 0.0f;
  }
}

__global__ void core_dtstrf_cmp_zzero_and_get_alpha_kernel(double* A, double* U, double zzero, double* dev_ptr)
{
  if((fabs(A[0]) == zzero) && (fabs(U[0]) == zzero))
  {
    dev_ptr[0] = 1.0f;
  }
  else
  {
    dev_ptr[0] = 0.0f;
  }
  
  dev_ptr[1] = ((double)1. / *U);
}

__global__ void core_dtstrf_set_zero_kernel(double* A, const int LDA,
                                            const int i, const int ii, const int im, const double zzero )
{
  const int j = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  
  if( j >= i )
    return;
  
  A[LDA*(ii+j)+im] = zzero;
}

#if defined(__cplusplus)
extern "C" {
#endif
  
#include <stdio.h>
  
void core_dtstrf_cmp(cudaStream_t stream, double* A, double* U, double* dev_ptr, double* host_ptr)
{
  dim3 threads( 1 );
  dim3 grid( 1 );
  
  core_dtstrf_cmp_kernel<<<threads, grid, 0, stream>>>(A, U, dev_ptr);
  cudaMemcpyAsync(host_ptr, dev_ptr, 2*sizeof(double), cudaMemcpyDeviceToHost, stream);
}

void core_dtstrf_cmp_zzero_and_get_alpha(cudaStream_t stream,
                           double* A, double* U, double zzero, double* dev_ptr, double* host_ptr)
{
  dim3 threads( 1 );
  dim3 grid( 1 );
  
  core_dtstrf_cmp_zzero_and_get_alpha_kernel<<<threads, grid, 0, stream>>>(A, U, zzero, dev_ptr);
  cudaMemcpyAsync(host_ptr, dev_ptr, 2*sizeof(double), cudaMemcpyDeviceToHost, stream);
}
  
void core_dtstrf_set_zero(cudaStream_t stream,
                          double* A, const int LDA,
                          const int i, const int ii, const int im, const double zzero )
{
  dim3 threads( BLOCK_SIZE );
  dim3 grid( (i+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);

  core_dtstrf_set_zero_kernel<<<threads, grid, 0, stream>>>(A, LDA, i, ii, im, zzero);
}
  
  
#if defined(__cplusplus)
}
#endif
