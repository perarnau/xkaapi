/**
 *
 * @file magmablas.h
 *
 *  PLASMA core_blas kernel for CUBLAS
 * (c) INRIA
 *
 * @author Joao Lima
 *
 * Based on MAGMA (version 1.1.0) http://icl.cs.utk.edu/magma
 *
 **/

#ifndef _MAGMABLAS_XKAAPI_H_
#define _MAGMABLAS_XKAAPI_H_

#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

void magmablas_dlaswp_kaapixx(cudaStream_t stream, int n, double *dAT, int lda,
                  int i1, int i2, int *ipiv, int inci);

void magmablas_dtranspose_kaapixx(cudaStream_t stream, double *odata, int ldo,
                     double *idata, int ldi, 
                     int m, int n );

void magmablas_dswapblk_kaapixx(cudaStream_t stream, int n,
                    double *dA1T, int lda1,
                    double *dA2T, int lda2,
                    int i1, int i2, int *ipiv, int inci, int offset);
  
void magmablas_dlacpy_kaapixx(cudaStream_t stream, char uplo, int m, int n,
                                double *a, int lda,
                              double *b, int ldb );
  
void magmablas_slaswp_kaapixx(cudaStream_t stream, int n, float *dAT, int lda,
                              int i1, int i2, int *ipiv, int inci);

void magmablas_stranspose_kaapixx(cudaStream_t stream, float *odata, int ldo,
                                  float *idata, int ldi,
                                  int m, int n );

void magmablas_sswapblk_kaapixx(cudaStream_t stream, int n,
                                float *dA1T, int lda1,
                                float *dA2T, int lda2,
                                int i1, int i2, int *ipiv, int inci, int offset);
  
#if defined(__cplusplus)
}
#endif

#endif /* _MAGMABLAS_XKAAPI_H_ */