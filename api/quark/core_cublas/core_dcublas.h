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
  
int CORE_dssssm_cublas(int M1, int N1, int M2, int N2, int K, int IB,
                       double *A1, int LDA1,
                       double *A2, int LDA2,
                       double *L1, int LDL1,
                       double *L2, int LDL2,
                       int *IPIV);
void CORE_dssssm_quark_cublas(Quark *quark);
  

#ifdef __cplusplus
}
#endif

#endif /* _CORE_CUBLAS_H_ */