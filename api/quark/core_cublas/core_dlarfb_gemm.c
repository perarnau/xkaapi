
#include "core_cublas.h"

/***************************************************************************//**
*
* @ingroup CORE_double
*
*  CORE_dlarfb_gemm applies a complex block reflector H or its transpose H'
*  to a complex M-by-N matrix C, from either the left or the right.
*  this kernel is similar to the lapack dlarfb but it do a full gemm on the
*  triangular Vs assuming that the upper part of Vs is zero and ones are on
*  the diagonal. It is also based on the fact that a gemm on a small block of
*  k reflectors is faster than a trmm on the triangular (k,k) + gemm below.
*
*  NOTE THAT:
*  Only Columnwise/Forward cases are treated here.
*
*******************************************************************************
*
* @param[in] side
*         @arg PlasmaLeft  : apply Q or Q**T from the Left;
*         @arg PlasmaRight : apply Q or Q**T from the Right.
*
* @param[in] trans
*         @arg PlasmaNoTrans   : No transpose, apply Q;
*         @arg PlasmaTrans : ConjTranspose, apply Q**T.
*
* @param[in] direct
*         Indicates how H is formed from a product of elementary
*         reflectors
*         @arg PlasmaForward  : H = H(1) H(2) . . . H(k) (Forward)
*         @arg PlasmaBackward : H = H(k) . . . H(2) H(1) (Backward)
*
* @param[in] storev
*         Indicates how the vectors which define the elementary
*         reflectors are stored:
*         @arg PlasmaColumnwise
*         @arg PlasmaRowwise
*
* @param[in] M
*         The number of rows of the matrix C.
*
* @param[in] N
*         The number of columns of the matrix C.
*
* @param[in] K
*          The order of the matrix T (= the number of elementary
*          reflectors whose product defines the block reflector).
*
* @param[in] V
*          COMPLEX*16 array, dimension
*              (LDV,K) if storev = 'C'
*              (LDV,M) if storev = 'R' and side = 'L'
*              (LDV,N) if storev = 'R' and side = 'R'
*          The matrix V. See further details.
*
* @param[in] LDV
*         The leading dimension of the array V.
*         If storev = 'C' and side = 'L', LDV >= max(1,M);
*         if storev = 'C' and side = 'R', LDV >= max(1,N);
*         if storev = 'R', LDV >= K.
*
* @param[in] T
*         The triangular K-by-K matrix T in the representation of the
*         block reflector.
*         T is upper triangular by block (economic storage);
*         The rest of the array is not referenced.
*
* @param[in] LDT
*         The leading dimension of the array T. LDT >= K.
*
* @param[in,out] C
*         COMPLEX*16 array, dimension (LDC,N)
*         On entry, the M-by-N matrix C.
*         On exit, C is overwritten by H*C or H'*C or C*H or C*H'.
*
* @param[in] LDC
*         The leading dimension of the array C. LDC >= max(1,M).
*
* @param[in,out] WORK
*         (workspace) COMPLEX*16 array, dimension (LDWORK,K).
*
* @param[in] LDWORK
*         The dimension of the array WORK.
*         If side = PlasmaLeft,  LDWORK >= max(1,N);
*         if side = PlasmaRight, LDWORK >= max(1,M).
*
*******************************************************************************
*
* @return
*          \retval PLASMA_SUCCESS successful exit
*          \retval <0 if -i, the i-th argument had an illegal value
*
******************************************************************************/

int CORE_dlarfb_gemm_cublas(PLASMA_enum side, PLASMA_enum trans, int direct, int storev,
                     int M, int N, int K,
                     const double *V, int LDV,
                     const double *T, int LDT,
                     double *C, int LDC,
                     double *WORK, int LDWORK)
{
  static double zzero =  0.0;
  static double zone  =  1.0;
  static double mzone = -1.0;
  cublasStatus_t status;  
#if defined(CONFIG_VERBOSE)
  fprintf(stdout, "%s: M=%d N=%d K=%d V=%p LDV=%d T=%p LDT=%d C=%p LDC=%d WORK=%p LDWORK=%d\n", __FUNCTION__,
          M, N, K, V, LDV, T, LDT, C, LDC, WORK, LDWORK
        );
  fflush(stdout);
#endif  
  
  /* Check input arguments */
  if ((side != PlasmaLeft) && (side != PlasmaRight)) {
    coreblas_error(1, "Illegal value of side");
    return -1;
  }
  if ((trans != PlasmaNoTrans) && (trans != PlasmaTrans)) {
    coreblas_error(2, "Illegal value of trans");
    return -2;
  }
  if ((direct != PlasmaForward) && (direct != PlasmaBackward)) {
    coreblas_error(3, "Illegal value of direct");
    return -3;
  }
  if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
    coreblas_error(4, "Illegal value of direct");
    return -4;
  }
  if (M < 0) {
    coreblas_error(5, "Illegal value of M");
    return -5;
  }
  if (N < 0) {
    coreblas_error(6, "Illegal value of N");
    return -6;
  }
  if (K < 0) {
    coreblas_error(7, "Illegal value of K");
    return -7;
  }
  
  /* Quick return */
  if ((M == 0) || (N == 0) || (K == 0) )
    return PLASMA_SUCCESS;
  
  /* For Left case, switch the trans. noswitch for right case */
  if( side == PlasmaLeft){
    if ( trans == PlasmaNoTrans) {
      trans = PlasmaTrans;
    }
    else {
      trans = PlasmaNoTrans;
    }
  }
  
  /* main code */
  if (storev == PlasmaColumnwise )
  {
    if ( direct == PlasmaForward )
    {
      /*
       * Let  V =  ( V1 )    (first K rows are unit Lower triangular)
       *           ( V2 )
       */
      if ( side == PlasmaLeft )
      {
        /*
         * Columnwise / Forward / Left
         */
        /*
         * Form  H * C  or  H' * C  where  C = ( C1 )
         *                                     ( C2 )
         *
         * W := C' * V    (stored in WORK)
         */
        status = cublasDgemm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToOp(CblasTrans),
                                PLASMA_CUBLAS_convertToOp(CblasNoTrans),
                                N, K, M,
                                &zone, C, LDC,
                                V, LDV,
                                &zzero, WORK, LDWORK );
        PLASMA_CUBLAS_ASSERT(status);
        
        /*
         * W := W * T'  or  W * T
         */
        status = cublasDtrmm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToSideMode(CblasRight),
                                PLASMA_CUBLAS_convertToFillMode(CblasUpper),
                                PLASMA_CUBLAS_convertToOp(trans),
                                PLASMA_CUBLAS_convertToDiagType(CblasNonUnit),
                                N, K,
                                &zone, T, LDT, WORK, LDWORK, WORK, LDWORK
                              );
        PLASMA_CUBLAS_ASSERT(status);
        
        /*
         * C := C - V * W'
         */
        status = cublasDgemm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToOp(CblasNoTrans),
                                PLASMA_CUBLAS_convertToOp(CblasTrans),
                                M, N, K,
                                &mzone, V, LDV,
                                WORK, LDWORK,
                                &zone, C, LDC
                              );
        PLASMA_CUBLAS_ASSERT(status);        
      }
      else {
        /*
         * Columnwise / Forward / Right
         */
        /*
         * Form  C * H  or  C * H'  where  C = ( C1  C2 )
         * W := C * V
         */
        status = cublasDgemm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToOp(CblasNoTrans),
                                PLASMA_CUBLAS_convertToOp(CblasNoTrans),
                                M, K, N,
                                &zone, C, LDC,
                                V, LDV,
                                &zzero, WORK, LDWORK);
        PLASMA_CUBLAS_ASSERT(status);
        
        /*
         * W := W * T  or  W * T'
         */
        status = cublasDtrmm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToSideMode(CblasRight),
                                PLASMA_CUBLAS_convertToFillMode(CblasUpper),
                                PLASMA_CUBLAS_convertToOp(trans),
                                PLASMA_CUBLAS_convertToDiagType(CblasNonUnit),
                                M, K,
                                &zone, T, LDT, WORK, LDWORK, WORK, LDWORK
                                );
        PLASMA_CUBLAS_ASSERT(status);
        
        /*
         * C := C - W * V'
         */
        status = cublasDgemm_v2( kaapi_cuda_cublas_handle(),
                                PLASMA_CUBLAS_convertToOp(CblasNoTrans),
                                PLASMA_CUBLAS_convertToOp(CblasTrans),
                                M, N, K,
                                &mzone, WORK, LDWORK,
                                V, LDV,
                                &zone, C, LDC
                                );
        PLASMA_CUBLAS_ASSERT(status);        
      }
    }
    else {
      /*
       * Columnwise / Backward / Left or Right
       */
      coreblas_error(3, "Not implemented (ColMajor / Backward / Left or Right)");
      return PLASMA_ERR_NOT_SUPPORTED;
    }
  }
  else {
    if (direct == PlasmaForward) {
      /*
       * Rowwise / Forward / Left or Right
       */
      coreblas_error(3, "Not implemented (RowMajor / Backward / Left or Right)");
      return PLASMA_ERR_NOT_SUPPORTED;
    }
    else {
      /*
       * Rowwise / Backward / Left or Right
       */
      coreblas_error(3, "Not implemented (RowMajor / Backward / Left or Right)");
      return PLASMA_ERR_NOT_SUPPORTED;
    }
  }
  return PLASMA_SUCCESS;
}
