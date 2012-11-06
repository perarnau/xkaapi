/**
 *
 * @file core_cublas.h
 *
 *  PLASMA core_dcublas kernel for CUBLAS
 * (c) INRIA
 *
 * @version 1.0
 * @author Joao Lima
 *
 **/

#ifndef _CORE_DCUBLAS_H_
#define _CORE_DCUBLAS_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "core_blas.h"

#ifdef __cplusplus
extern "C" {
#endif
  
void CORE_dgemm_cublas(int transA, int transB,
                       int m, int n, int k,
                       double alpha, const double *A, int lda,
                       const double *B, int ldb,
                       double beta, double *C, int ldc);
void CORE_dgemm_quark_cublas(Quark *quark);
void CORE_dgemm_f2_quark_cublas(Quark* quark);
void CORE_dgemm_p2_quark_cublas(Quark* quark);
void CORE_dgemm_p2f1_quark_cublas(Quark* quark);
void CORE_dgemm_p3_quark_cublas(Quark* quark);

void CORE_dsyrk_cublas(int uplo, int trans,
                       int n, int k,
                       double alpha, const double *A, int LDA,
                       double beta, double *C, int LDC);
void CORE_dsyrk_quark_cublas(Quark *quark);
  
void CORE_dtrsm_cublas(int side, int uplo,
                       int transA, int diag,
                       int M, int N,
                       double alpha, const double *A, int LDA,
                       double *B, int LDB);
void CORE_dtrsm_quark_cublas(Quark *quark);

#if 0
/***************************************************************************
 *   register CUBLAS equiv. functions
 **/
CUBLAS_QUARK_FUNCTION(CORE_dplgsy_quark, 0, CUQUARK_CPU_ONLY);

/* dgemm related */
CUBLAS_QUARK_FUNCTION(CORE_dgemm_quark, CORE_dgemm_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dgemm_f2_quark, CORE_dgemm_f2_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dgemm_p2_quark, CORE_dgemm_p2_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dgemm_p3_quark, CORE_dgemm_p3_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dgemm_p2f1_quark, CORE_dgemm_p2f1_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dsyrk_quark, CORE_dsyrk_quark_cublas, CUQUARK_GPU_ONLY);
CUBLAS_QUARK_FUNCTION(CORE_dtrsm_quark, CORE_dtrsm_quark_cublas, CUQUARK_GPU_ONLY);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _CORE_CUBLAS_H_ */