/*
 ** xkaapi
 **
 ** Copyright 2009, 2010, 2011, 2012 INRIA.
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

#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <math.h>
#include "kaapi++" // this is the new C++ interface for Kaapi

#if defined(CONFIG_USE_DOUBLE)
typedef double double_type;
#elif defined(CONFIG_USE_FLOAT)
typedef float double_type;
#endif



#include "../matrix/matrix.h"

static int BLOCSIZE = 0;

// check results
#define CONFIG_DO_CHECK 1
#if CONFIG_DO_CHECK

# include <stdlib.h>

template<typename T>
static int do_check(const T* a, const T* b, T* c, T* c_old, unsigned int n)
{
  unsigned int i, j, k;
  T* const tmp = c_old;

    CBLAS<T>::gemm
    (
	CblasColMajor, CblasNoTrans, CblasNoTrans,
	n, n, n, 1.0, a, n, b, n, 1.0, tmp, n
    );

  int res = -1;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      k = i * n + j;
      if (fabsf(c[k] - tmp[k]) >= 0.01)
      {
	printf("ERROR invalid @%lx,%u,%u %f != %f\n", (uintptr_t)c, i, j, c[k], tmp[k]);
	goto on_error;
      }
    }
  }

  res = 0;

 on_error:
  return res;
}

#endif // CONFIG_DO_CHECK

template<typename T>
struct TaskParallelGEMM: public ka::Task<8>::Signature
<
  CBLAS_ORDER,			      /* row / col */
  CBLAS_TRANSPOSE,        /* NoTrans/Trans for A */
  CBLAS_TRANSPOSE,        /* NoTrans/Trans for B */
  T,                      /* alpha */
  ka::R<ka::range2d<T> >, /* Aik   */
  ka::R<ka::range2d<T> >, /* Akj   */
  T,                      /* beta */
  ka::RPWP<ka::range2d<T> > /* Aij   */
>{};

#define TASK_GEMM_THRESHOLD	  512
template<typename T>
struct TaskBodyCPU<TaskParallelGEMM<T> > {
  void operator()
  (
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    T alpha,
    ka::range2d_r<T> A,
    ka::range2d_r<T> B,
    T beta,
    ka::range2d_rpwp<T> C
  )
  {
    const T* const a = A->ptr();
    const T* const b = B->ptr();
    T* const c       = C->ptr();

    const size_t M = A->dim(0);
    const size_t K = B->dim(0);
    const size_t N = B->dim(1);

    const int lda = A->lda();
    const int ldb = B->lda();
    const int ldc = C->lda();
    const int bloc = TASK_GEMM_THRESHOLD;

#if CONFIG_VERBOSE
    fprintf(stdout, "TaskCPU ParallelGEMM m=%d n=%d k=%d A=%p alpha=%.2f B=%p beta=%.2f C=%p lda=%d ldb=%d ldc=%d\n", M, N, K, (void*)a, alpha, (void*)b, beta, (void*)c, lda, ldb, ldc ); fflush(stdout);
#endif
    if( M > TASK_GEMM_THRESHOLD )
    {
      for (size_t i=0; i<M; i += bloc)
      {
	ka::rangeindex ri(i, i+bloc);
	for (size_t j=0; j<N; j += bloc)
	{
	  ka::rangeindex rj(j, j+bloc);
	  for (size_t k=0; k<K; k += bloc)
	  {
	    ka::rangeindex rk(k, k+bloc);
	      ka::Spawn<TaskGEMM<T> >(ka::SetArch(ka::ArchHost))
		(
		 order, transA, transB,
		 alpha, A(rk,ri), B(rj,rk), beta, C(rj,ri)
		);
	  }
	}
      }
    } else {
      // spawn here
      CBLAS<T>::gemm
      (
	order, transA, transB,
	M, N, K, alpha, a, lda, b, ldb, beta, c, ldc
      );
    }
  }
};

template<typename T>
struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<T> >, /* A */
      ka::R<ka::range2d<T> >,  /* B */
      ka::RPWP<ka::range2d<T> >   /* C */
>{};

template<typename T>
struct TaskBodyCPU<TaskMatProduct<T> > {
  void operator()( ka::range2d_r<T> A, ka::range2d_r<T> B, ka::range2d_rpwp<T> C )
  {
    size_t M = A->dim(0);
    size_t K = B->dim(0);
    size_t N = B->dim(1);
    int bloc = BLOCSIZE;
    
    for (size_t i=0; i<M; i += bloc)
    {
      ka::rangeindex ri(i, i+bloc);
      for (size_t j=0; j<N; j += bloc)
      {
	ka::rangeindex rj(j, j+bloc);
        for (size_t k=0; k<K; k += bloc)
        {
          ka::rangeindex rk(k, k+bloc);
            ka::Spawn<TaskParallelGEMM<T> >(ka::SetStaticSched())
	      (  CblasColMajor, CblasNoTrans, CblasNoTrans, (T)1.0, 
		   B(rk,ri), A(rj,rk), (T)1.0, C(rj,ri) );
        }
      }
    }
  }
};

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv)
  {
    // av[1] = matrix_size
    // av[2] = block_size
    int matrix_size = 512;
    int block_size = 1;
    int verif = 0;

    if( argc > 1 )
	    matrix_size= atoi(argv[1]);
    if( argc > 2 )
	    block_size= atoi(argv[2]);
    if( argc > 3 )
	    verif = 1; // check results ON

    BLOCSIZE = block_size;

    const int n = matrix_size;

    double_type* dA = (double_type*) malloc(n * n * sizeof(double_type));
    double_type* dB = (double_type*) malloc(n * n * sizeof(double_type));
    double_type* dC = (double_type*) malloc(n * n * sizeof(double_type));
#if CONFIG_DO_CHECK
    double_type* dC_old;
#endif
    if (0 == dA || 0 == dB || 0 == dC) 
    {
        std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
            << std::endl;
        return;
    }

    ka::array<2,double_type> A( dA, n, n, n);
    ka::array<2,double_type> B( dB, n, n, n);
    ka::array<2,double_type> C( dC, n, n, n);

    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(A) );
    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(B) );
    TaskBodyCPU<TaskLARNV<double_type> >()( ka::range2d_w<double_type>(C) );

    /* register memory to Xkaapi runtime */
#if CONFIG_USE_CUDA
    ka::Memory::Register( A );
    ka::Memory::Register( B );
    ka::Memory::Register( C );
#endif

#if CONFIG_DO_CHECK
    if( verif ){
	dC_old= (double_type*) calloc(n* n, sizeof(double_type));
	if( dC_old == 0){
        std::cout << "Fatal Error. Cannot allocate auxiliary matrix C."
            << std::endl;
        return;
	}
	memcpy( dC_old, dC, n*n*sizeof(double_type) );
    }
#endif

    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct<double_type> >(ka::SetStaticSched())( A, B, C );
    ka::Sync();
#if CONFIG_USE_CUDA
    ka::MemorySync();
#endif

    // dont time memory sync for the benchmarks since
    // it does not reflect the execution pipeline
    double t1 = kaapi_get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * n * n * n)/(t1-t0));

    fprintf( stdout, "# size bloc threads time GFlop/s\n" );
    fprintf( stdout, "GEMM %d %d %d %.10f %.6f\n", matrix_size, block_size,
		    kaapi_getconcurrency(), tdelta, gflops );
    fflush(stdout);

#if CONFIG_DO_CHECK
    if( verif ){
	    if( do_check(dA, dB, dC, dC_old, n) == -1 )
	      fprintf(stdout, "# ERROR invalid matrix\n");
	    else
		    fprintf(stdout, "# output OK\n");
	    fflush(stdout);
	    free(dC_old);
    }
#endif

#if CONFIG_USE_CUDA
    ka::Memory::Unregister( A );
    ka::Memory::Unregister( B );
    ka::Memory::Unregister( C );
#endif
    free(dA);
    free(dB);
    free(dC);
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }

  return 0;
}

