/**
 *
 * @file cublas_core_dtrsm.c
 *
 *  PLASMA core_blas kernel for CUBLAS
 * (c) INRIA
 *
 * @version 2.4.6
 * @author Thierry Gautier
 *
 **/

#include "core_cublas.h"

void CORE_dtrsm_cublas(int side, int uplo,
                int transA, int diag,
                int M, int N,
                double alpha, const double *A, int LDA,
                double *B, int LDB)
{
  cublasStatus_t status = cublasDtrsm_v2(
         kaapi_cuda_cublas_handle(),
         side,
         uplo,
         transA,
         diag,
         M, N,
         &(alpha), A, LDA,
         B, LDB);
  
  PLASMA_CUBLAS_ASSERT(status);
#if 0
  fprintf(stdout,"%s: a=%p b=%p m=%d n=%d\n", __FUNCTION__,
          A, B, M, N);
  fflush(stdout);
#endif
}

/***************************************************************************
 *
 **/
void CORE_dtrsm_quark_cublas(Quark *quark)
{
    PLASMA_enum side;
    PLASMA_enum uplo;
    PLASMA_enum transA;
    PLASMA_enum diag;
    int m;
    int n;
    double alpha;
    double *A;
    int lda;
    double *B;
    int ldb;

    quark_unpack_args_11(quark, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
    CORE_dtrsm_cublas(
      PLASMA_CUBLAS_convertToSideMode(side),
      PLASMA_CUBLAS_convertToFillMode(uplo),
      PLASMA_CUBLAS_convertToOp(transA),
      PLASMA_CUBLAS_convertToDiagType(diag),
      m, n,
      (alpha), A, lda,
      B, ldb);
}
