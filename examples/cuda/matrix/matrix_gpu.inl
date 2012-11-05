/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
** Joao.Lima@imag.fr / joao.lima@inf.ufrgs.br
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

// needed for flags
#include <cblas.h>

#if CONFIG_USE_CUBLAS
#include "cublas_v2.h"
#else
#include "cublas.h"
#endif

#if (CONFIG_USE_CUBLAS) || (CONFIG_USE_MAGMA)

#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif
  extern cublasHandle_t kaapi_cuda_cublas_handle( void );
#if defined(__cplusplus)
}
#endif

#endif

#if defined(CONFIG_USE_CUBLAS)
#include "cublas.inl"
#endif

#if defined(CONFIG_USE_MAGMA)
#include "magma.inl"
#endif

// task definitions

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
    
    const int m = Aik->dim(1);
    const int n = Aik->dim(0); // eq. to Akj->rows();
    const int k = Akj->dim(0);
    
    
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
struct TaskBodyGPU<TaskGEMM2<T> >
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
   ka::range2d_rw<T> Aij,
   ka::pointer_w<int> fake
   )
  {
    const T* const a = Aik->ptr();
    const T* const b = Akj->ptr();
    T* const c       = Aij->ptr();
    
    const int m = Aik->dim(1);
    const int n = Aik->dim(0); // eq. to Akj->rows();
    const int k = Akj->dim(0);
    
    
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
    const int n     = A->dim(1);
    const int k     = A->dim(0); // eq. to Akj->rows();
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
    
    const int n = C->dim(1);
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
template<typename T>
struct TaskBodyGPU<TaskGETRF<T> > {
  void operator()(
                  ka::gpuStream stream,
                  CBLAS_ORDER order,
                  ka::range2d_rw<T> A,
                  uintptr_t piv
                  )
  {
    const int m        = A->dim(0);
    const int n        = A->dim(1);
    const int lda      = A->lda();
    T* const a    = A->ptr();
    int* const ipiv = (int*)piv;
    
#if defined(CONFIG_VERBOSE)
    fprintf(stdout, "TaskGPU TaskGETRF m=%d n=%d lda=%d A=%p Piv=%p\n",
            m, n, lda, (void*)a, ipiv ); fflush(stdout);
#endif
    
    const int info = MAGMA<T>::getrf( m, n, a, lda, ipiv );
    if (info){
      fprintf( stdout, "TaskDGETRF::magma_getrf() ERROR %d\n", info );
      fflush(stdout);
    }
  }
};
#endif

#if defined(CONFIG_USE_MAGMA)
template<typename T>
struct TaskBodyGPU<TaskGETF2NoPiv<T> > {
  void operator()(
                  ka::gpuStream stream,
                  CBLAS_ORDER order,
                  ka::range2d_rw<T> A
                  )
  {
    const int m        = A->dim(0);
    const int n        = A->dim(1);
    const int lda      = A->lda();
    T* const a    = A->ptr();
    
    const int info = MAGMA<T>::getrf_nopiv( m, n, a, lda );
    if (info){
      fprintf( stdout, "TaskDGETRF::magma_getrf_nopiv() ERROR %d\n", info );
      fflush(stdout);
    }
  }
};
#endif


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
    int info = 0;
    const char uplo_ = (uplo == CblasUpper) ? 'U' : 'L';
    info = MAGMA<T>::potrf( uplo_, n, a, lda );
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


#endif /* ! MATRIX_GPU_INL_INCLUDED */
