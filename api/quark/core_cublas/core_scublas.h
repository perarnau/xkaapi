/**
 *
 * @file core_scublas.h
 *
 *  PLASMA core_scublas kernel for CUBLAS
 * (c) INRIA
 *
 * @version 1.0
 * @author Joao Lima
 *
 **/

#ifndef _CORE_SCUBLAS_H_
#define _CORE_SCUBLAS_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "core_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

void CORE_sgemm_cublas(int transA, int transB,
                       int M, int N, int K,
                       float alpha, float *A, int LDA,
                       float *B, int LDB,
                       float beta, float *C, int LDC);
void CORE_sgemm_quark_cublas(Quark *quark);
  
#ifdef __cplusplus
}
#endif

#endif