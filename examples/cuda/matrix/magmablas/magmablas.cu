
#ifndef BLOCK_SIZE
#define BLOCK_SIZE  64
#endif

#include <cuda.h>
#include "magmablas.h"

typedef struct {
        double *A;
        int n, lda, j0, npivots;
        short ipiv[BLOCK_SIZE];
} dlaswp_params_t2;

__global__ void mydlaswp2( dlaswp_params_t2 params )
{
        unsigned int tid = threadIdx.x + __mul24(blockDim.x, blockIdx.x);
        if( tid < params.n )
        {
                int lda = params.lda;
                double *A = params.A + tid + lda * params.j0;

                for( int i = 0; i < params.npivots; i++ )
                {
                         int j = params.ipiv[i];
                        double *p1 = A + i*lda;
                        double *p2 = A + j*lda;
                        double temp = *p1;
                        *p1 = *p2;
                        *p2 = temp;
                }
        }
}

#if defined(__cplusplus)
extern "C" {
#endif

void dlaswp3( cudaStream_t stream, dlaswp_params_t2 &params )
{
         int blocksize = 64;
        dim3 blocks = (params.n+blocksize-1) / blocksize;
        mydlaswp2<<< blocks, blocksize, 0, stream >>>( params );
}

void 
magmablas_dlaswp( cudaStream_t stream, int n, double *dAT, int lda, 
                  int i1, int i2, int *ipiv, int inci )
{
  int k;
  
  for( k=(i1-1); k<i2; k+=BLOCK_SIZE )
    {
      int sb = min(BLOCK_SIZE, i2-k);
      //dlaswp_params_t params = { dAT, lda, lda, ind + k };
      dlaswp_params_t2 params = { dAT+k*lda, n, lda, 0, sb };
      for( int j = 0; j < sb; j++ )
        {
          params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
      dlaswp3( stream, params );
    }
}

#if defined(__cplusplus)
}
#endif
