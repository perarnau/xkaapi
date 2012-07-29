/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/

#ifndef MATRIX_GPU_INL_INCLUDED
# define MATRIX_GPU_INL_INCLUDED


#include "kaapi++"
#include <stdio.h>

#include <cuda_runtime_api.h>

// needed for flags
#include <cblas.h>


#if CONFIG_USE_CUBLAS
#include "cublas_v2.h"
#else
#include "cublas.h"
#endif

#if 0
#include "magma.h"
#include "magmablas.h"
#endif

#if (CONFIG_USE_CUBLAS) || (CONFIG_USE_MAGMA)

#if defined(__cplusplus)
extern "C" {
#endif
extern cublasHandle_t kaapi_cuda_cublas_handle( void );
#if defined(__cplusplus)
}
#endif

#endif


#if CONFIG_USE_CUBLAS
/* from cublas.h */

/* Helper functions */
static inline cublasOperation_t convertToOp( int trans ) 
//static inline cublasOperation_t convertToOp( const enum CBLAS_TRANSPOSE trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return CUBLAS_OP_N;
        case CblasTrans:
            return CUBLAS_OP_T;
        case CblasConjTrans:
            return CUBLAS_OP_C;                        
        default:
            return CUBLAS_OP_N;
    }

}

//static inline cublasFillMode_t convertToFillMode( const enum CBLAS_UPLO uplo ) 
static inline cublasFillMode_t convertToFillMode( int uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return CUBLAS_FILL_MODE_UPPER;
	case CblasLower:
        default:
         return CUBLAS_FILL_MODE_LOWER;
    }        
}

//static inline cublasDiagType_t convertToDiagType( const enum CBLAS_DIAG diag ) 
static inline cublasDiagType_t convertToDiagType( int diag ) 
{
    switch (diag) {
	case CblasUnit:
            return CUBLAS_DIAG_UNIT;
	case CblasNonUnit:
        default:
         return CUBLAS_DIAG_NON_UNIT;
    }        
}

//static inline cublasSideMode_t convertToSideMode( const enum CBLAS_SIDE side ) 
static inline cublasSideMode_t convertToSideMode( int side ) 
{
    switch (side) {
	case CblasRight:
            return CUBLAS_SIDE_RIGHT;
	case CblasLeft:
        default:
         return CUBLAS_SIDE_LEFT;
    }        
}

#endif /* CONFIG_USE_CUBLAS */

#if 0

/* Old CUBLAS API, returns char parameters */

//static inline char convertToOp( const enum CBLAS_TRANSPOSE trans ) 
static inline char convertToOp( int trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return 'n';
        case CblasTrans:
            return 't';
        case CblasConjTrans:
            return 'c';                        
        default:
            return 'n';
    }

}

//static inline char convertToFillMode( const enum CBLAS_UPLO uplo ) 
static inline char convertToFillMode( int uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return 'u';
	case CblasLower:
        default:
         return 'l';
    }        
}

//static inline char convertToDiagType( const enum CBLAS_DIAG diag ) 
static inline char convertToDiagType( int diag ) 
{
    switch (diag) {
	case CblasUnit:
            return 'u';
	case CblasNonUnit:
        default:
         return 'n';
    }        
}

//static inline char convertToSideMode( const enum CBLAS_SIDE side ) 
static inline char convertToSideMode( int side ) 
{
    switch (side) {
	case CblasRight:
            return 'r';
	case CblasLeft:
        default:
         return 'l';
    }        
}

#endif /* CONFIG_USE_MAGMA */


/* for cublas v2 */
template<class T>
struct CUBLAS {
  typedef T value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                     cublasSideMode_t side,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     cublasDiagType_t diag,
                     int m,
                     int n,
                     const value_type *alpha, /* host or device pointer */
                     const value_type *A,
                     int lda,
                     value_type *B,
                     int ldb);
                                         
  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc);
  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc);
};

template<>
struct CUBLAS<double> {
  typedef double value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc)
  { return cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); }
};

template<>
struct CUBLAS<float> {
  typedef float value_type;
  static cublasStatus_t trsm(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t trans,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const value_type *alpha, /* host or device pointer */
                   const value_type *A,
                   int lda,
                   value_type *B,
                   int ldb)
  { return cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); }

  static cublasStatus_t gemm( cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    const value_type *alpha, /* host or device pointer */
                    const value_type *A,
                    int lda,
                    const value_type *B,
                    int ldb,
                    const value_type *beta, /* host or device pointer */
                    value_type *C,
                    int ldc)
  { return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }

  static cublasStatus_t syrk(cublasHandle_t handle,
                     cublasFillMode_t uplo,
                     cublasOperation_t trans,
                     int n,
                     int k,
                     const value_type *alpha,  /* host or device pointer */
                     const value_type *A,
                     int lda,
                     const value_type *beta,  /* host or device pointer */
                     value_type *C,
                     int ldc)
  { return cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); }
};


#if CONFIG_USE_MAGMA
typedef int magma_int_t;

#if defined(__cplusplus)
extern "C" {
#endif

magma_int_t magma_spotrf_gpu
(char, magma_int_t, float*, magma_int_t, magma_int_t*);

magma_int_t magma_dpotrf_gpu
(char, magma_int_t, double*, magma_int_t, magma_int_t*);

magma_int_t magma_sgetrf_gpu
(
 magma_int_t m, magma_int_t n, float *A,
 magma_int_t lda, magma_int_t *ipiv,
 magma_int_t *info
);

magma_int_t magma_dgetrf_gpu
(
 magma_int_t m, magma_int_t n, double *A,
 magma_int_t lda, magma_int_t *ipiv,
 magma_int_t *info
);

magma_int_t magma_sgetrf_nopiv_gpu
(
 magma_int_t m, magma_int_t n, float *A,
 magma_int_t lda, 
 magma_int_t *info
);

magma_int_t magma_dgetrf_nopiv_gpu
(
 magma_int_t m, magma_int_t n, double *A,
 magma_int_t lda, 
 magma_int_t *info
);

void magmablas_sgemm(char tA, char tB,
		     magma_int_t m, magma_int_t n, magma_int_t k, 
		     float alpha,
		     const float *A, magma_int_t lda, 
		     const float *B, magma_int_t ldb, 
		     float beta,
		     float *C, magma_int_t ldc);

void magmablas_dgemm(char tA, char tB,
		     magma_int_t m, magma_int_t n, magma_int_t k, 
		     double alpha,
		     const double *A, magma_int_t lda, 
		     const double *B, magma_int_t ldb, 
		     double beta,
		     double *C, magma_int_t ldc);

magma_int_t
magma_dgessm_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib, 
                  magma_int_t *ipiv, 
                  double *dL1, magma_int_t lddl1, 
                  double *dL,  magma_int_t lddl, 
                  double *dA,  magma_int_t ldda, 
                  magma_int_t *info);

magma_int_t
magma_dtstrf_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                  double *hU, magma_int_t ldhu, double *dU, magma_int_t lddu, 
                  double *hA, magma_int_t ldha, double *dA, magma_int_t ldda, 
                  double *hL, magma_int_t ldhl, double *dL, magma_int_t lddl,
                  magma_int_t *ipiv, 
                  double *hwork, magma_int_t ldhwork, double *dwork, magma_int_t lddwork,
                  magma_int_t *info);

magma_int_t
magma_dssssm_gpu(char storev, magma_int_t m1, magma_int_t n1, 
                 magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib, 
                 double *dA1, magma_int_t ldda1, 
                 double *dA2, magma_int_t ldda2, 
                 double *dL1, magma_int_t lddl1, 
                 double *dL2, magma_int_t lddl2,
                 magma_int_t *IPIV, magma_int_t *info);


void   magmablas_dlaswp( magma_int_t N, 
             double *dAT, magma_int_t lda, 
             magma_int_t i1,  magma_int_t i2, 
             magma_int_t *ipiv, magma_int_t inci );


magma_int_t magma_dgeqrf_gpu( magma_int_t m, magma_int_t n, 
                              double *dA,  magma_int_t ldda, 
                              double *tau, double *dT, 
                              magma_int_t *info);

magma_int_t magma_dormqr_gpu( char side, char trans, 
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              double *a,    magma_int_t lda, double *tau, 
                              double *c,    magma_int_t ldc,
                              double *work, magma_int_t lwork, 
                              double *td,   magma_int_t nb, magma_int_t *info);

#if defined(__cplusplus)
}
#endif

#if CONFIG_USE_FLOAT

# define magma_potrf magma_spotrf_gpu
# define magma_getrf magma_sgetrf_gpu
# define magma_getrf_nopiv magma_sgetrf_nopiv_gpu
#define magmablas_gemm magmablas_sgemm

#else

# define magma_potrf magma_dpotrf_gpu
# define magma_getrf magma_dgetrf_gpu
# define magma_getrf_nopiv magma_dgetrf_nopiv_gpu
#define magmablas_gemm magmablas_dgemm

#endif

#endif // CONFIG_USE_MAGMA

// task definitions

template<typename T> 
struct TaskBodyGPU<TaskTRSM_left<T> >
{
  void operator()
  (
    ka::gpuStream     stream,
    ka::range2d_r<T>  Akk,
    ka::range2d_rw<T> Akj
  )
  {
#if CONFIG_USE_CUBLAS
    const T* const a = Akk->ptr();
    const int lda = Akk->lda();

    T* const b = Akj->ptr();
    const int ldb   = Akj->lda();
    const int n     = Akj->dim(0);
    const int m     = Akj->dim(1);

    static const T static_alpha = 1.;

    const cublasStatus_t status = CUBLAS<T>::trsm
      (
       kaapi_cuda_cublas_handle(),
       convertToSideMode(CblasLeft),
       convertToFillMode(CblasLower),
       convertToOp(CblasNoTrans),
       convertToDiagType(CblasUnit),
       n, m, &static_alpha, a, lda, b, ldb
      );

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasTrsm() == %d\n", __FUNCTION__, status);
#endif
  }
};


template<typename T> 
struct TaskBodyGPU<TaskTRSM_right<T> >
{
  void operator()
  (
    ka::gpuStream stream,
    ka::range2d_r<T> Akk,
    ka::range2d_rw<T> Aik
  )
  {
#if CONFIG_USE_CUBLAS
    const T* const a = Akk->ptr();
    const int lda = Akk->lda();

    T* const b = Aik->ptr();
    const int ldb = Aik->lda();
    const int n = Aik->dim(0); // b.rows();
    const int m = Aik->dim(1); // b.cols();

    static const T static_alpha = 1.;

    const cublasStatus_t status = CUBLAS<T>::trsm
      (
       kaapi_cuda_cublas_handle(),
       convertToSideMode(CblasRight),
       convertToFillMode(CblasUpper),
       convertToOp(CblasNoTrans),
       convertToDiagType(CblasNonUnit),
       n, m,
       &static_alpha, a, lda,
       b, ldb
      );

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasTrsm() == %d\n", __FUNCTION__, status);
#endif
  }
};

template<typename T> 
struct TaskBodyGPU<TaskGEMM<T> >
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   CBLAS_TRANSPOSE transA,
   CBLAS_TRANSPOSE transB,
   T alpha,
   ka::range2d_r<T> Aik,
   ka::range2d_r<T> Akj,
   T beta,
   ka::range2d_rw<T> Aij
  )
  {
    const T* const a = Aik->ptr();
    const T* const b = Akj->ptr();
    T* const c       = Aij->ptr();

    const int m = Aik->dim(0); 
    const int n = Aik->dim(1); // eq. to Akj->rows();
    const int k = Akj->dim(1); 


    const int lda = Aik->lda();
    const int ldb = Akj->lda();
    const int ldc = Aij->lda();


#if 0
    fprintf(stdout, "TaskGPU GEMM m=%d n=%d k=%d A=%p alpha=%.2f B=%p beta=%.2f C=%p lda=%d ldb=%d ldc=%d\n", m, n, k, (void*)a, alpha, (void*)b, beta, (void*)c, lda, ldb, ldc ); fflush(stdout);
#endif

    KAAPI_TIMING_CUDA_BEGIN((cudaStream_t)stream.stream);
#if CONFIG_USE_CUBLAS
	cublasStatus_t status;
	if( order == CblasColMajor ) 
		status = CUBLAS<T>::gemm (
		       kaapi_cuda_cublas_handle(),
		       convertToOp(transA),
		       convertToOp(transB),
		       m, n, k,
		       &alpha,
		       a, lda,
		       b, ldb, 
		       &beta, c, ldc
    );
	else {
    kaapi_assert_debug( order == CblasRowMajor )
		status = CUBLAS<T>::gemm(
		       kaapi_cuda_cublas_handle(),
		       convertToOp(transB),
		       convertToOp(transA),
		       n, m, k,
		       &alpha,
		       b, ldb, 
		       a, lda,
		       &beta, c, ldc
    );
  }
	
  if (status != CUBLAS_STATUS_SUCCESS)
    printf("%s::cublasGemm() == %d\n", __FUNCTION__, status);

#elif CONFIG_USE_MAGMA
	if( order == CblasColMajor ) 
		magmablas_gemm(
			//MagmaNoTrans,
			convertToOp(transA),
			//MagmaNoTrans,
			convertToOp(transB),
			m, n, k,
			alpha,
			a, lda,
			b, ldb,
			beta,
			c, ldc
		);
	else if( order == CblasRowMajor )
		magmablas_gemm(
			//MagmaNoTrans,
			convertToOp(transB),
			//MagmaNoTrans,
			convertToOp(transA),
			n, m, k,
			alpha,
			b, ldb,
			a, lda,
			beta,
			c, ldc
		);
#endif
    KAAPI_TIMING_CUDA_END((cudaStream_t)stream.stream, "GPU DGEMM", n);
  }
};


template<typename T> 
struct TaskBodyGPU<TaskSYRK<T> >
{
  void operator()
  (
    ka::gpuStream stream,
    CBLAS_ORDER		   order, 
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    T alpha,
    ka::range2d_r <T>  A, 
    T beta,
    ka::range2d_rw<T> C 
  )
  {
    const int n     = A->dim(0); 
    const int k     = A->dim(1); // eq. to Akj->rows();
    const int lda   = A->lda();
    const T* const a = A->ptr();

    const int ldc   = C->lda();
    T* const c = C->ptr();

#if 0
    fprintf(stdout, "TaskGPU DSYRK n=%d k=%d lda=%d A=%p ldc=%d C=%p\n",
		n, k, lda, (void*)a, ldc, c ); fflush(stdout);
#endif

    KAAPI_TIMING_CUDA_BEGIN((cudaStream_t)stream.stream);
#if CONFIG_USE_CUBLAS
    const cublasStatus_t status = CUBLAS<T>::syrk
      (
       kaapi_cuda_cublas_handle(),
       convertToFillMode(uplo),
       convertToOp(trans),
       n, k,
       &alpha, a, lda,
       &beta,
       c, ldc
      );

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasSyrk() == %d\n", __FUNCTION__, status);
#endif
    KAAPI_TIMING_CUDA_END((cudaStream_t)stream.stream, "GPU DSYRK", n);
  }
};


template<typename T> 
struct TaskBodyGPU<TaskTRSM<T> >
{
  void operator()
  (
    ka::gpuStream	  stream,
    CBLAS_ORDER		   order, 
    CBLAS_SIDE             side,
    CBLAS_UPLO             uplo,
    CBLAS_TRANSPOSE        transA,
    CBLAS_DIAG             diag,
    T                 alpha,
    ka::range2d_r <T> A, 
    ka::range2d_rw<T> C
  )
  {
    const T* const a = A->ptr();
    const int lda = A->lda();

    T* const c = C->ptr();
    const int ldc = C->lda();

    const int n = C->dim(0);
    const int k = (transA == CblasNoTrans ? A->dim(1) : A->dim(0) );

#if 0
    fprintf(stdout, "TaskGPU DTRSM n=%d k=%d lda=%d A=%p ldc=%d C=%p\n",
		n, k, lda, (void*)a, ldc, c ); fflush(stdout);
#endif

    KAAPI_TIMING_CUDA_BEGIN((cudaStream_t)stream.stream);
#if CONFIG_USE_CUBLAS
    const cublasStatus_t status = CUBLAS<T>::trsm
      (
       kaapi_cuda_cublas_handle(),
       convertToSideMode(side),
       convertToFillMode(uplo),
       convertToOp(transA),
       convertToDiagType(diag),
       n, k,
       &alpha, a, lda,
       c, ldc
      );

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasTrsm() == %d\n", __FUNCTION__, status);

#endif
    KAAPI_TIMING_CUDA_END((cudaStream_t)stream.stream, "GPU DTRSM", n);
  }
};

#if defined(CONFIG_USE_MAGMA)
template<>
struct TaskBodyGPU<TaskDGETRF> {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range2d_rw<double> A, 
    ka::range1d_w<int> piv
  )
  {
    const int m        = A->dim(0); 
    const int n        = A->dim(1); 
    const int lda      = A->lda();
    double* const a    = A->ptr();
    int* const ipiv = piv->ptr();

#if 1
    fprintf(stdout, "TaskGPU DGETRF m=%d n=%d lda=%d A=%p Piv=%p\n",
		m, n, lda, (void*)a, ipiv ); fflush(stdout);
#endif

    int* const hipiv = (int*) calloc( piv->size(), sizeof(int) );

    magma_int_t info = 0;
    magma_dgetrf_gpu( m, n, a, lda, hipiv, &info );
    if (info){
	fprintf( stdout, "TaskDGETRF::magma_getrf() ERROR %d\n", info );
	fflush(stdout);
    }
    /* TODO */
    cudaMemcpyAsync( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyHostToDevice,
	    (cudaStream_t)stream.stream
     );
  }
};
#endif

template<typename T> 
struct TaskBodyGPU<TaskGETRFNoPiv<T> >
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   ka::range2d_rw<T> A
  )
  {
    const int m        = A->dim(0); 
    const int n        = A->dim(1); 
//    const int lda      = A->lda();
    const int lda      = A->dim(1);
    T* const a    = A->ptr();

#if KAAPI_VERBOSE
    fprintf(stdout, "TaskGPU DGETRF m=%d n=%d lda=%d A=%p\n",
		m, n, lda, (void*)a ); fflush(stdout);
#endif

#if CONFIG_USE_MAGMA
    magma_int_t info = 0;
    magma_getrf_nopiv( m, n, a, lda, &info );
    if (info){
	fprintf( stdout, "TaskDGETRF::magma_getrf_nopiv() ERROR %d\n", info );
	fflush(stdout);
	}
#else
    const int ione   = 1;
    int* piv = (int*) calloc(m, sizeof(int));
    T* work = (T*) calloc( n*lda, sizeof(T));

    cudaMemcpy2D( work, lda*sizeof(T), a,
	    lda*sizeof(T), n, lda, 
	    cudaMemcpyDeviceToHost );
    CLAPACK<T>::getrf( order, m, n, work, lda, piv );
    LAPACKE<T>::laswp( order, m, work, lda, ione, n, piv, ione);
    cudaMemcpy2D( a, lda*sizeof(T), work,
	    lda*sizeof(T), n, lda,
	    cudaMemcpyHostToDevice );
    free( piv );
    free( work );
#endif
  }
};


#if 0 /* DO NOT USE GPU FOR POTRF */
template<typename T> 
struct TaskBodyGPU<TaskPOTRF<T> >
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   CBLAS_UPLO uplo,
   ka::range2d_rw<T> A
  )
  {
    const int n     = A->dim(0); 
    const int lda   = A->lda();
    T* const a = A->ptr();

#if 1
    fprintf(stdout, "TaskGPU DPOTRF m=%d A=%p lda=%d\n", n, (void*)a, lda ); fflush(stdout);
#endif
#if CONFIG_USE_MAGMA
    magma_int_t info = 0;
    const char uplo_ = (uplo == CblasUpper) ? 'U' : 'L';
    magma_potrf( uplo_, n, a, lda, &info );
    if (info){
	fprintf( stdout, "TaskDPOTRF::magma_potrf() ERROR %d\n", info );
	fflush(stdout);
	}
#else
    T* work = (T*) calloc( n*lda, sizeof(T));
    cudaMemcpy2D( work, lda*sizeof(T),
	    a, lda*sizeof(T),
	    lda*sizeof(T), n, 
	    cudaMemcpyDeviceToHost );
    CLAPACK<T>::potrf( order,  uplo, n, work, lda );
    cudaMemcpy2D( a, lda*sizeof(T),
	    work, lda*sizeof(T),
	    lda*sizeof(T), n,
	    cudaMemcpyHostToDevice );
    free( work );
#endif

  }
};
#endif

#if defined(CONFIG_USE_DOUBLE)
template<>
struct TaskBodyGPU<TaskPlasmaDSSSSM> {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range2d_rw<double> A1, 
    ka::range2d_rw<double> A2,
    ka::range2d_r<double> L1,
    ka::range2d_r<double> L2,
    ka::range1d_r<int> piv
  )
  {
    int m1 = A1->dim(0); 
    int n1 = A1->dim(1);
    int m2 = A2->dim(0); 
    int n2 = A2->dim(1);
    int k = L1->dim(0);
    int lda1 = A1->lda();
    int lda2 = A2->lda();
    int ldl1 = L1->lda();
    int ldl2 = L2->lda();
    double* const a1 = A1->ptr();
    double* const a2 = A2->ptr();
    double* const l1 = (double*)L1->ptr();
    double* const l2 = (double*)L2->ptr();
    int* const ipiv = (int*)piv->ptr();
    const int ib = IB; // from PLASMA

#if 0
    fprintf( stdout, "TaskGPU DSSSSM A1(%lu,%lu) a1=%p lda1=%d "
	   "A2(%lu,%lu) a2=%p lda2=%d "
	   "L1(%lu,%lu) l1=%p ldl1=%d "
	   "L2(%lu,%lu) l2=%p ldl2=%d ipiv=%p\n",
	   A1->dim(0), A1->dim(1), (void*)a1, lda1,
	   A2->dim(0), A2->dim(1), (void*)a2, lda2,
	   L1->dim(0), L1->dim(1), (void*)l1, ldl1,
	   L2->dim(0), L2->dim(1), (void*)l2, ldl2,
	   ipiv
	);
    fflush(stdout);
#endif
#if 0
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    magma_int_t info = 0;
    magma_dssssm_gpu(
	    'f',
	    m1, n1, m2, n2, k, ib,
	    a1, lda1,
	    a2, lda2,
	    l1, ldl1,
	    l2, ldl2,
	    hipiv,
	    &info
    );
#endif

#if 1
    static double zone  = 1.0;
    static double mzone =-1.0;
    cublasStatus_t status;

    int* h_ipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( h_ipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    int i, ii, sb;
    int im, ip;
    ip = 0;

    for(ii = 0; ii < k; ii += ib) {
        sb = std::min(k-ii, ib);

        for(i = 0; i < sb; i++) {
            im = h_ipiv[ip]-1;

            if (im != (ii+i)) {
                im = im - m1;
		status = cublasDswap
		  (
		   kaapi_cuda_cublas_handle(),
		   n1,
		   &a1[ii+i], lda1,
		   &a2[im], lda2
		  );
            }
            ip = ip + 1;
        }

	status = CUBLAS<double>::trsm
	  (
	   kaapi_cuda_cublas_handle(),
	   convertToSideMode(CblasLeft),
	   convertToFillMode(CblasLower),
	   convertToOp(CblasNoTrans),
	   convertToDiagType(CblasUnit),
	   sb, n1, &zone,
           &l1[ldl1*ii], ldl1,
           &a1[ii], lda1
	  );
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("TaskPlasmaDSSSSM::cublasTrsm() == %d\n", status);

	status = CUBLAS<double>::gemm
	(
	    kaapi_cuda_cublas_handle(),
	    convertToOp(CblasNoTrans),
	    convertToOp(CblasNoTrans),
	    m2, n2, sb,
	    &mzone,
	    &l2[ldl2*ii], ldl2,
            &a1[ii], lda1,
            &zone, a2, lda2
	);
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("TaskPlasmaDSSSSM::cublasGemm() == %d\n", status);
    }
#endif
  }
};
#endif

#if defined(CONFIG_USE_MAGMA)
template<>
struct TaskBodyGPU<TaskPlasmaDGESSM> {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<double> L, 
    ka::range2d_rw<double> A
  )
  {
    int m = A->dim(0); 
    int n = A->dim(1);
    int k = A->dim(1);
    int lda = A->lda();
    int ldl = L->lda();
    double* const a = A->ptr();
    double* const l = (double*)L->ptr();
    int* const ipiv = (int*)piv->ptr();
    const int ib = IB; // from PLASMA

#if 0
    fprintf( stdout, "TaskGPU DGESSM A(%lu,%lu) a=%p lda=%d  L(%lu,%lu) l=%p ldl=%d\n",
	    A->dim(0), A->dim(1), (void*)a, A->lda(),
	    L->dim(0), L->dim(1), (void*)l, L->lda()
	   );
    fflush(stdout);
#endif

    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    magma_int_t info = 0;
    magma_dgessm_gpu( 
	'f',
	m, n, k, ib,
	hipiv,
	l, ldl,
	a, lda,
	a, lda,
	&info
    );
#if 0
    int* h_ipiv = (int*)calloc( k, sizeof(int) );
    cudaMemcpy( h_ipiv, ipiv, k * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    static double zone  =  1.0;
    static double mzone = -1.0;
    static int                ione  =  1;
    cublasStatus_t status;

    int i, sb;
    int tmp, tmp2;

    for(i = 0; i < k; i += ib) {
        sb = std::min(ib, k-i);
        /*
         * Apply interchanges to columns I*IB+1:IB*( I+1 )+1.
         */
        tmp  = i+1;
        tmp2 = i+sb;
	magmablas_dlaswp(
	    n,
	    a, lda,
	    tmp, tmp2,
	    h_ipiv, ione
	);

	/* magmablas_dlaswp */
        /*
         * Compute block row of U.
         */
	status = cublasDtrsm (
	   kaapi_cuda_cublas_handle(),
	   convertToSideMode(CblasLeft),
	   convertToFillMode(CblasLower),
	   convertToOp(CblasNoTrans),
	   convertToDiagType(CblasUnit),
	   sb, n,
	   &zone,
           &l[ldl*i+i], ldl,
           &a[i], lda
	  );
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("DGESSM::cublasDtrsm() == %d\n", status);

        if (i+sb < m) {
	    /*
	    * Update trailing submatrix.
	    */
	    status = cublasDgemm (
		kaapi_cuda_cublas_handle(),
		convertToOp(CblasNoTrans), convertToOp(CblasNoTrans),
		m-(i+sb), n, sb,
		&mzone,
		&l[ldl*i+(i+sb)], ldl,
		&a[i], lda,
		&zone, &a[i+sb], lda
	    );
	    if (status != CUBLAS_STATUS_SUCCESS)
	      printf("DGESSM::cublasDgemm() == %d\n", status);
	}
    }
#endif
  }
};
#endif

#if defined(CONFIG_USE_MAGMA)
template<>
struct TaskBodyGPU<TaskPlasmaDTSTRF> {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<double> U, 
    ka::range2d_rw<double> A,
    ka::range2d_rw<double> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<double> WORK
  )
  {
    int m = A->dim(0); 
    int n = A->dim(1);
    int lda = A->lda();
    int ldl = L->lda();
    int ldu = U->lda();
    int ldw = WORK->lda();
    double* const a = A->ptr();
    double* const l = L->ptr();
    double* const u = U->ptr();
    double* const work = WORK->ptr();
    int* const ipiv = piv->ptr();
    const int ib = IB; // from PLASMA

#if 0
    fprintf( stdout, "TaskGPU DTSTRF A(%lu,%lu) a=%p lda=%d "
	    "L(%lu,%lu) l=%p ldl=%d "
	    "U(%lu,%lu) u=%p ldu=%d ipiv=%p\n",
	    A->dim(0), A->dim(1), (void*)a, lda,
	    L->dim(0), L->dim(1), (void*)l, ldl,
	    U->dim(0), U->dim(1), (void*)u, ldu,
	    (void*)ipiv 
    );
    fflush(stdout);
#endif

    double* ha = (double*)malloc( A->dim(0)*A->dim(1) * sizeof(double) );
    double* hl = (double*)malloc( L->dim(0)*L->dim(1) * sizeof(double) );
    double* hu = (double*)malloc( U->dim(0)*U->dim(1) * sizeof(double) );
    double* hwork = (double*)malloc( WORK->dim(0)*WORK->dim(1) * sizeof(double) );
    cublasGetMatrix( m, n, sizeof(double), u, ldu, hu, ldu );
    cublasGetMatrix( m, ib, sizeof(double), a, lda, ha, lda );
    memset( hl, 0, L->dim(0)*L->dim(1)*sizeof(double) );
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    magma_int_t info = 0;
    magma_dtstrf_gpu( 
	'f',
	m, n, ib, L->dim(1),
	hu, ldu, u, ldu,
	ha, lda, a, lda,
	hl, ldl, l, ldl,
	hipiv,
	hwork, ldw, work, lda,
	&info
    );
    /* TODO */
    cudaMemcpyAsync( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyHostToDevice,
	    (cudaStream_t)stream.stream
     );
  }
};
#endif

#endif /* ! MATRIX_GPU_INL_INCLUDED */
