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

#if CONFIG_USE_MAGMA
#include "magma.h"
#include "magmablas.h"
#endif

#if CONFIG_USE_VOLKOV
#include "volkov_sgemm.cu"
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
static inline cublasOperation_t convertToOp( const enum CBLAS_TRANSPOSE trans ) 
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
static inline cublasFillMode_t convertToFillMode( const enum CBLAS_UPLO uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return CUBLAS_FILL_MODE_UPPER;
	case CblasLower:
        default:
         return CUBLAS_FILL_MODE_LOWER;
    }        
}

static inline cublasDiagType_t convertToDiagType( const enum CBLAS_DIAG diag ) 
{
    switch (diag) {
	case CblasUnit:
            return CUBLAS_DIAG_UNIT;
	case CblasNonUnit:
        default:
         return CUBLAS_DIAG_NON_UNIT;
    }        
}

static inline cublasSideMode_t convertToSideMode( const enum CBLAS_SIDE side ) 
{
    switch (side) {
	case CblasRight:
            return CUBLAS_SIDE_RIGHT;
	case CblasLeft:
        default:
         return CUBLAS_SIDE_LEFT;
    }        
}

/* CONFIG_USE_CUBLAS */
#elif CONFIG_USE_MAGMA

/* Old CUBLAS API, returns char parameters */

static inline char convertToOp( const enum CBLAS_TRANSPOSE trans ) 
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
static inline char convertToFillMode( const enum CBLAS_UPLO uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return 'u';
	case CblasLower:
        default:
         return 'l';
    }        
}

static inline char convertToDiagType( const enum CBLAS_DIAG diag ) 
{
    switch (diag) {
	case CblasUnit:
            return 'u';
	case CblasNonUnit:
        default:
         return 'n';
    }        
}

static inline char convertToSideMode( const enum CBLAS_SIDE side ) 
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

#if CONFIG_USE_FLOAT
# define cublasTrsm cublasStrsm
# define cublasGemm cublasSgemm
# define cublasSyrk cublasSsyrk
#else
# define cublasTrsm cublasDtrsm
# define cublasGemm cublasDgemm
# define cublasSyrk cublasDsyrk
#endif // CONFIG_USE_FLOAT

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

template<> struct TaskBodyGPU<TaskDTRSM_left>
{
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_r<double_type> Akk,
   ka::range2d_rw<double_type> Akj
  )
  {
#if CONFIG_USE_CUBLAS
    const double_type* const a = Akk.ptr();
    const int lda = Akk.lda();

    double_type* const b = Akj.ptr();
    const int ldb   = Akj.lda();
    const int n     = Akj.dim(0);
    const int m     = Akj.dim(1);

    static const double_type static_alpha = 1.;

    const cublasStatus_t status = cublasTrsm
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


template<> struct TaskBodyGPU<TaskDTRSM_right>
{
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_r<double_type> Akk,
   ka::range2d_rw<double_type> Aik
  )
  {
#if CONFIG_USE_CUBLAS

    const double_type* const a = Akk.ptr();
    const int lda = Akk.lda();

    double_type* const b = Aik.ptr();
    const int ldb = Aik.lda();
    const int n = Aik.dim(0); // b.rows();
    const int m = Aik.dim(1); // b.cols();

    static const double_type static_alpha = 1.;

    const cublasStatus_t status = cublasTrsm
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

template<> struct TaskBodyGPU<TaskDGEMM>
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   CBLAS_TRANSPOSE transA,
   CBLAS_TRANSPOSE transB,
   double_type alpha,
   ka::range2d_r<double_type> Aik,
   ka::range2d_r<double_type> Akj,
   double_type beta,
   ka::range2d_rw<double_type> Aij
  )
  {
    const double_type* const a = Aik.ptr();
    const double_type* const b = Akj.ptr();
    double_type* const c       = Aij.ptr();

    const int m = Aik.dim(0); 
    const int n = Aik.dim(1); // eq. to Akj.rows();
    const int k = Akj.dim(1); 


    const int lda = Aik.dim(1); 
    const int ldb = Aik.dim(1);
    const int ldc = Akj.dim(1); 
    //const int lda = Aik.lda();
    //const int ldb = Akj.lda();
    //const int ldc = Aij.lda();


#if 0
    fprintf(stdout, "TaskGPU GEMM m=%d n=%d k=%d A=%p alpha=%.2f B=%p beta=%.2f C=%p lda=%d ldb=%d ldc=%d\n", m, n, k, (void*)a, alpha, (void*)b, beta, (void*)c, lda, ldb, ldc ); fflush(stdout);
#endif

#if CONFIG_USE_CUBLAS
	cublasStatus_t status;
	if( order == CblasColMajor ) 
		status = cublasGemm
		      (
		       kaapi_cuda_cublas_handle(),
		       convertToOp(transA),
		       convertToOp(transB),
		       m, n, k,
		       &alpha,
		       a, lda,
		       b, ldb, 
		       &beta, c, ldc
		      );
	else if( order == CblasRowMajor )
		status = cublasGemm
		      (
		       kaapi_cuda_cublas_handle(),
		       convertToOp(transB),
		       convertToOp(transA),
		       n, m, k,
		       &alpha,
		       b, ldb, 
		       a, lda,
		       &beta, c, ldc
		      );
	
    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasGemm() == %d\n", __FUNCTION__, status);

#elif CONFIG_USE_MAGMA
	if( order == CblasColMajor ) 
		magmablas_gemm(
			MagmaNoTrans,
			//convertToOp(transA),
			MagmaNoTrans,
			//convertToOp(transB),
			m, n, k,
			alpha,
			a, lda,
			b, ldb,
			beta,
			c, ldc
		);
	else if( order == CblasRowMajor )
		magmablas_gemm(
			MagmaNoTrans,
			//convertToOp(transA),
			MagmaNoTrans,
			//convertToOp(transB),
			m, n, k,
			alpha,
			b, ldb,
			a, lda,
			beta,
			c, ldc
		);
#endif
#if 0
    volkov_sgemm
    (
     (CUstream)stream.stream,
     c, a, b, lda, ldb, ldc
    );
    size_t mm = m * m;
    const int mnk = m;
    const size_t thread_count = mm < 512 ? mm : 512;
    mulKernel<<<1, dim3(thread_count), 0, 0>>>
      (a, b, c, m);
#endif
  }
};


template<> struct TaskBodyGPU<TaskDSYRK>
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   CBLAS_UPLO uplo,
   CBLAS_TRANSPOSE trans,
   double_type alpha,
   ka::range2d_r <double_type>  A, 
   double_type beta,
   ka::range2d_rw<double_type> C 
  )
  {
    const int n     = A.dim(0); 
    const int k     = A.dim(1); // eq. to Akj.rows();
    const int lda   = A.dim(1);
    //const int lda   = A.lda();
    const double_type* const a = A.ptr();

    //const int ldc   = C.lda();
    const int ldc   = C.dim(1);
    double_type* const c = C.ptr();

#if KAAPI_VERBOSE
    fprintf(stdout, "TaskGPU DSYRK n=%d k=%d lda=%d A=%p ldc=%d C=%p\n",
		n, k, lda, (void*)a, ldc, c ); fflush(stdout);
#endif

#if CONFIG_USE_CUBLAS
    const cublasStatus_t status = cublasSyrk
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
#elif CONFIG_USE_MAGMA
	/* MAGMA uses the old cublas interface */
    cublasSyrk
      (
       convertToFillMode(uplo),
       convertToOp(trans),
       n, k,
       alpha, a, lda,
       beta,
       c, ldc
      );
	const cublasStatus_t status = cublasGetError();

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasSyrk() == %d\n", __FUNCTION__, status);
#endif
  }
};


template<> struct TaskBodyGPU<TaskDTRSM>
{
  void operator()
  (
   ka::gpuStream	  stream,
   CBLAS_ORDER		   order, 
   CBLAS_SIDE             side,
   CBLAS_UPLO             uplo,
   CBLAS_TRANSPOSE        transA,
   CBLAS_DIAG             diag,
   double_type                 alpha,
   ka::range2d_r <double_type> A, 
   ka::range2d_rw<double_type> C
  )
  {
    const double_type* const a = A.ptr();
    const int lda = A.dim(1);

    double_type* const c = C.ptr();
    const int ldc = C.dim(1);

    const int n = C.dim(0);
    const int k = (transA == CblasNoTrans ? A.dim(1) : A.dim(0) );

#if KAAPI_VERBOSE
    fprintf(stdout, "TaskGPU DTRSM n=%d k=%d lda=%d A=%p ldc=%d C=%p\n",
		n, k, lda, (void*)a, ldc, c ); fflush(stdout);
#endif

#if CONFIG_USE_CUBLAS
    const cublasStatus_t status = cublasTrsm
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

#elif CONFIG_USE_MAGMA
	/* MAGMA uses the old cublas interface */
    cublasTrsm
      (
       convertToSideMode(side),
       convertToFillMode(uplo),
       convertToOp(transA),
       convertToDiagType(diag),
       n, k,
       alpha, a, lda,
       c, ldc
      );
	const cublasStatus_t status = cublasGetError();

    if (status != CUBLAS_STATUS_SUCCESS)
      printf("%s::cublasTrsm() == %d\n", __FUNCTION__, status);
#endif
  }
};


template<> struct TaskBodyGPU<TaskDGETRF>
{
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_rw<double_type> A, 
   ka::range1d_w<int> piv
  )
  {
#if CONFIG_USE_MAGMA
    const int m        = A.dim(0); 
    const int n        = A.dim(1); 
//    const int lda      = A.lda();
    const int lda      = A.dim(1);
    double_type* const a    = A.ptr();

#if KAAPI_VERBOSE
    int* const ipiv = piv.ptr();
    fprintf(stdout, "TaskGPU DGETRF m=%d n=%d lda=%d A=%p Piv=%p\n",
		m, n, lda, (void*)a, ipiv ); fflush(stdout);
#endif

	/* TODO: MAGMA assumes that a(from A) is in GPU and ipiv is in CPU */
    int* const h_ipiv = (int*) calloc( piv.size(), sizeof(int) );

    magma_int_t info = 0;
    magma_getrf( m, n, a, lda, h_ipiv, &info );
    if (info){
	fprintf( stdout, "TaskDGETRF::magma_getrf() ERROR %d\n", info );
	fflush(stdout);
	}
#endif
  }

};

template<> struct TaskBodyGPU<TaskDGETRFNoPiv>
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   ka::range2d_rw<double_type> A
  )
  {
    const int m        = A.dim(0); 
    const int n        = A.dim(1); 
//    const int lda      = A.lda();
    const int lda      = A.dim(1);
    double_type* const a    = A.ptr();

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
    double_type* work = (double_type*) calloc( n*lda, sizeof(double_type));

    cudaMemcpy2D( work, lda*sizeof(double_type), a,
	    lda*sizeof(double_type), n, lda, 
	    cudaMemcpyDeviceToHost );
    clapack_getrf( order, m, n, work, lda, piv );
    LAPACKE_laswp( order, m, work, lda, ione, n, piv, ione);
    cudaMemcpy2D( a, lda*sizeof(double_type), work,
	    lda*sizeof(double_type), n, lda,
	    cudaMemcpyHostToDevice );
    free( piv );
    free( work );
#endif
  }
};


template<> struct TaskBodyGPU<TaskDPOTRF>
{
  void operator()
  (
   ka::gpuStream stream,
   CBLAS_ORDER		   order, 
   CBLAS_UPLO uplo,
   ka::range2d_rw<double_type> A
  )
  {
    const int n     = A.dim(0); 
    //const int lda   = A.lda();
    const int lda   = A.dim(1);
    double_type* const a = A.ptr();

#if KAAPI_VERBOSE
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
    double_type* work = (double_type*) calloc( n*lda, sizeof(double_type));
    cudaMemcpy2D( work, lda*sizeof(double_type), a,
	    lda*sizeof(double_type), n, lda, 
	    cudaMemcpyDeviceToHost );
    clapack_potrf( order,  uplo, n, work, lda );
    cudaMemcpy2D( a, lda*sizeof(double_type), work,
	    lda*sizeof(double_type), n, lda,
	    cudaMemcpyHostToDevice );
    free( work );
#endif

  }
};


#endif /* ! MATRIX_GPU_INL_INCLUDED */
