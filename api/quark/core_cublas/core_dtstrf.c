#include "core_cublas.h"

extern cudaStream_t kaapi_cuda_kernel_stream(void);

extern void core_dtstrf_cmp_zzero_and_get_alpha(cudaStream_t stream,
                                         double* A, double* U, double zzero, double* dev_ptr, double* host_ptr);
extern void core_dtstrf_cmp(cudaStream_t stream, double* A, double* U, double* dev_ptr, double* host_ptr);
extern void core_dtstrf_set_zero(cudaStream_t stream,
                          double* A, const int LDA,
                          const int i, const int ii, const int im, const double zzero );

int CORE_dtstrf_cublas(int M, int N, int IB, int NB,
                double *U, int LDU,
                double *A, int LDA,
                double *L, int LDL,
                int *IPIV,
                double *WORK, int LDWORK,
                int *INFO)
{
  static double zzero = 0.0;
  static double mzone =-1.0;
  cublasStatus_t status;
  cudaError_t err;
  
  double alpha;
  int i, j, ii, sb;
  int im, ip;
  
#if CONFIG_VERBOSE
  fprintf(stdout, "%s: M=%d N=%d IB=%d NB=%d U=%p LDU=%d A=%p LDA=%d L=%p LDL=%d IPIV=%p WORK=%p LDWORK=%d\n",
          __FUNCTION__, M, N, IB, NB, U, LDU, A, LDA, L, LDL, IPIV, WORK, LDWORK);
  fflush(stdout);
#endif
  
  /* Check input arguments */
  *INFO = 0;
  if (M < 0) {
    coreblas_error(1, "Illegal value of M");
    return -1;
  }
  if (N < 0) {
    coreblas_error(2, "Illegal value of N");
    return -2;
  }
  if (IB < 0) {
    coreblas_error(3, "Illegal value of IB");
    return -3;
  }
  if ((LDU < max(1,NB)) && (NB > 0)) {
    coreblas_error(6, "Illegal value of LDU");
    return -6;
  }
  if ((LDA < max(1,M)) && (M > 0)) {
    coreblas_error(8, "Illegal value of LDA");
    return -8;
  }
  if ((LDL < max(1,IB)) && (IB > 0)) {
    coreblas_error(10, "Illegal value of LDL");
    return -10;
  }
  
  /* Quick return */
  if ((M == 0) || (N == 0) || (IB == 0))
    return PLASMA_SUCCESS;
  
  /* Set L to 0 */
  err = cudaMemset(L, 0, LDL*N*sizeof(double));
  PLASMA_CUDA_ASSERT(err);
  
  double* dev_ptr = 0;
  err = cudaMalloc((void**)&dev_ptr, 2*sizeof(double));
  PLASMA_CUDA_ASSERT(err);
  double* host_ptr;
  err = cudaMallocHost((void**)&host_ptr, 2*sizeof(double));
  PLASMA_CUDA_ASSERT(err);
  
  int* piv = kaapi_memory_get_host_pointer_and_validate(IPIV);
  
  ip = 0;
  for (ii = 0; ii < N; ii += IB) {
    sb = min(N-ii, IB);
    
    for (i = 0; i < sb; i++) {
      status = cublasIdamax(kaapi_cuda_cublas_handle(),
                            M, &A[LDA*(ii+i)], 1, &im
                            );
      PLASMA_CUBLAS_ASSERT(status);
      
      /* get im */
      err = cudaStreamSynchronize(kaapi_cuda_kernel_stream());
      PLASMA_CUDA_ASSERT(err);
      
      /* ajust index, CUBLAS is 1-based indexing */
      im--;

      piv[ip] = ii+i+1;
      
      core_dtstrf_cmp(kaapi_cuda_kernel_stream(),
                      &A[LDA*(ii+i)+im], &U[LDU*(ii+i)+ii+i], dev_ptr, host_ptr);
      err = cudaStreamSynchronize(kaapi_cuda_kernel_stream());
      PLASMA_CUDA_ASSERT(err);
      
      if (host_ptr[0] == 1.0f) {
        /*
         * Swap behind.
         */
        status = cublasDswap(kaapi_cuda_cublas_handle(),
                   i, &L[LDL*ii+i], LDL, &WORK[im], LDWORK
        );
        PLASMA_CUBLAS_ASSERT(status);
        /*
         * Swap ahead.
         */
        status = cublasDswap(kaapi_cuda_cublas_handle(),
              sb-i, &U[LDU*(ii+i)+ii+i], LDU, &A[LDA*(ii+i)+im], LDA
         );
        PLASMA_CUBLAS_ASSERT(status);
        /*
         * Set IPIV.
         */
        piv[ip] = NB + im + 1;

        core_dtstrf_set_zero(kaapi_cuda_kernel_stream(),
                             A, LDA, i, ii, im, zzero
                        );
      }
      
      core_dtstrf_cmp_zzero_and_get_alpha(kaapi_cuda_kernel_stream(),
                      &A[LDA*(ii+i)+im], &U[LDU*(ii+i)+ii+i], zzero, dev_ptr, host_ptr);
      err = cudaStreamSynchronize(kaapi_cuda_kernel_stream());
      PLASMA_CUDA_ASSERT(err);
      
      if ((*INFO == 0) && (host_ptr[0] == 1.0f)) {
        *INFO = ii+i+1;
      }
      
//      alpha = ((double)1. / U[LDU*(ii+i)+ii+i]);
      alpha = host_ptr[1];
      status = cublasDscal(kaapi_cuda_cublas_handle(),
                           M, &alpha, &A[LDA*(ii+i)], 1
                           );
      PLASMA_CUBLAS_ASSERT(status);
      
      status = cublasDcopy(kaapi_cuda_cublas_handle(),
                  M, &A[LDA*(ii+i)], 1, &WORK[LDWORK*i], 1
        );
      PLASMA_CUBLAS_ASSERT(status);
      
      status = cublasDger(kaapi_cuda_cublas_handle(),
                          M, sb-i-1,
                          &mzone, &A[LDA*(ii+i)], 1,
                          &U[LDU*(ii+i+1)+ii+i], LDU,
                          &A[LDA*(ii+i+1)], LDA
      );
      PLASMA_CUBLAS_ASSERT(status);
      ip = ip+1;
    }
    /*
     * Apply the subpanel to the rest of the panel.
     */
    if(ii+i < N) {
      for(j = ii; j < ii+sb; j++) {
        if (piv[j] <= NB) {
          piv[j] = piv[j] - ii;
        }
      }
      
      CORE_dssssm_cublas_v2(
                  NB, N-(ii+sb), M, N-(ii+sb), sb, sb,
                  &U[LDU*(ii+sb)+ii], LDU,
                  &A[LDA*(ii+sb)], LDA,
                  &L[LDL*ii], LDL,
                  WORK, LDWORK, &piv[ii]
                  );
      err = cudaStreamSynchronize(kaapi_cuda_kernel_stream());
      PLASMA_CUDA_ASSERT(err);
      
      for(j = ii; j < ii+sb; j++) {
        if (piv[j] <= NB) {
          piv[j] = piv[j] + ii;
        }
      }
    }
  }
  
  cudaFreeHost(host_ptr);
  cudaFree(dev_ptr);
  return PLASMA_SUCCESS;
}

void CORE_dtstrf_quark_cublas(Quark *quark)
{
  int m;
  int n;
  int ib;
  int nb;
  double *U;
  int ldu;
  double *A;
  int lda;
  double *L;
  int ldl;
  int *IPIV;
  double *WORK;
  int ldwork;
  PLASMA_sequence *sequence;
  PLASMA_request *request;
  PLASMA_bool check_info;
  int iinfo;
  
  int info;
  
  quark_unpack_args_17(quark, m, n, ib, nb, U, ldu, A, lda, L, ldl, IPIV, WORK, ldwork, sequence, request, check_info, iinfo);
  CORE_dtstrf_cublas(m, n, ib, nb, U, ldu, A, lda, L, ldl, IPIV, WORK, ldwork, &info);
  if (info != PLASMA_SUCCESS && check_info)
    plasma_sequence_flush(quark, sequence, request, iinfo + info);
}