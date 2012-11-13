/**
 *
 * @file cublas_core_dgemm.c
 *
 *  PLASMA core_blas kernel for CUBLAS
 * (c) INRIA
 *
 * @version 2.4.6
 * @author Thierry Gautier
 *
 **/

#include "core_cublas.h"

void CORE_dgemm_cublas(int transA, int transB,
                int m, int n, int k,
                double alpha, const double *A, int lda,
                const double *B, int ldb,
                double beta, double *C, int ldc)
{
  cublasStatus_t status = cublasDgemm_v2(
                                         kaapi_cuda_cublas_handle(),
                                         transA,
                                         transB,
                                         m, n, k,
                                         &alpha, A, lda,
                                         B, ldb,
                                         &beta, C, ldc);
  PLASMA_CUBLAS_ASSERT(status);
#if CONFIG_VERBOSE
  fprintf(stdout,"%s: a=%p b=%p c=%p m=%d n=%d k=%d\n", __FUNCTION__,
          A, B, C, m, n, k);
  fflush(stdout);
#endif
}

/***************************************************************************//**
 *   CUBLAS version of CORE_dgemm_quark
 **/
void CORE_dgemm_quark_cublas(Quark *quark)
{
    PLASMA_enum transA;
    PLASMA_enum transB;
    int m;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double *B;
    int ldb;
    double beta;
    double *C;
    int ldc;

    quark_unpack_args_13(quark, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  
    CORE_dgemm_cublas(
      PLASMA_CUBLAS_convertToOp(transA),
      PLASMA_CUBLAS_convertToOp(transB),
      m, n, k,
      alpha, A, lda,
      B, ldb,
      beta, C, ldc);
}


/***************************************************************************//**
 *
 **/
void CORE_dgemm_f2_quark_cublas(Quark* quark)
{
    PLASMA_enum transA;
    PLASMA_enum transB;
    int M;
    int N;
    int K;
    double alpha;
    double *A;
    int LDA;
    double *B;
    int LDB;
    double beta;
    double *C;
    int LDC;
    void *fake1, *fake2;

    quark_unpack_args_15(quark, transA, transB, M, N, K, alpha,
                         A, LDA, B, LDB, beta, C, LDC, fake1, fake2);

    CORE_dgemm_cublas(
        PLASMA_CUBLAS_convertToOp(transA),
        PLASMA_CUBLAS_convertToOp(transB),
        M, N, K,
        alpha, A, LDA,
        B, LDB,
        beta, C, LDC);
}


/***************************************************************************//**
 *
 **/
void CORE_dgemm_p2_quark_cublas(Quark* quark)
{
    PLASMA_enum transA;
    PLASMA_enum transB;
    int M;
    int N;
    int K;
    double alpha;
    double *A;
    int LDA;
    double **B;
    int LDB;
    double beta;
    double *C;
    int LDC;

    quark_unpack_args_13(quark, transA, transB, M, N, K, alpha,
                         A, LDA, B, LDB, beta, C, LDC);

    CORE_dgemm_cublas(
                    PLASMA_CUBLAS_convertToOp(transA),
                    PLASMA_CUBLAS_convertToOp(transB),
                    M, N, K,
                    alpha, A, LDA,
                    *B, LDB,
                    beta, C, LDC);
}

/***************************************************************************//**
 *
 **/
void CORE_dgemm_p3_quark_cublas(Quark* quark)
{
    PLASMA_enum transA;
    PLASMA_enum transB;
    int M;
    int N;
    int K;
    double alpha;
    double *A;
    int LDA;
    double *B;
    int LDB;
    double beta;
    double **C;
    int LDC;

    quark_unpack_args_13(quark, transA, transB, M, N, K, alpha,
                         A, LDA, B, LDB, beta, C, LDC);

    CORE_dgemm_cublas(
                    PLASMA_CUBLAS_convertToOp(transA),
                    PLASMA_CUBLAS_convertToOp(transB),
                    M, N, K,
                    alpha, A, LDA,
                    B, LDB,
                    beta, *C, LDC);
}


/***************************************************************************//**
 *
 **/
void CORE_dgemm_p2f1_quark_cublas(Quark* quark)
{
    PLASMA_enum transA;
    PLASMA_enum transB;
    int M;
    int N;
    int K;
    double alpha;
    double *A;
    int LDA;
    double **B;
    int LDB;
    double beta;
    double *C;
    int LDC;
    void *fake1;

    quark_unpack_args_14(quark, transA, transB, M, N, K, alpha,
                         A, LDA, B, LDB, beta, C, LDC, fake1);

    CORE_dgemm_cublas(
                      PLASMA_CUBLAS_convertToOp(transA),
                      PLASMA_CUBLAS_convertToOp(transB),
                      M, N, K,
                      alpha, A, LDA,
                      *B, LDB,
                      beta, C, LDC);
}
