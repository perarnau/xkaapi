/*
 * matrix-mulitpy.cilk
 *
 * An implementation of matrix multiply based on Cilk parallelization (matrix_multiply.cilk)
 * but using Kaapi C++ construction
 
 * First of five matrix multiply examples to compare dense matrix multiplication
 * algorithms using Cilk parallelization.
 *   Example 1: Straightforward loop parallelization of matrix multiplication.
 *
 * Copyright (c) 2007-2008 Cilk Arts, Inc.  55 Cambridge Street,
 * Burlington, MA 01803.  Patents pending.  All rights reserved. You may
 * freely use the sample code to guide development of your own works,
 * provided that you reproduce this notice in any works you make that
 * use the sample code.  This sample code is provided "AS IS" without
 * warranty of any kind, either express or implied, including but not
 * limited to any implied warranty of non-infringement, merchantability
 * or fitness for a particular purpose.  In no event shall Cilk Arts,
 * Inc. be liable for any direct, indirect, special, or consequential
 * damages, or any other damages whatsoever, for any use of or reliance
 * on this sample code, including, without limitation, any lost
 * opportunity, lost profits, business interruption, loss of programs or
 * data, even if expressly advised of or otherwise aware of the
 * possibility of such damages, whether in an action of contract,
 * negligence, tort, or otherwise.
 *
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <math.h>
#include "kaapi++" // this is the new C++ interface for Kaapi

typedef double double_type;

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
    
    for (size_t j=0; j<M; j += bloc)
    {
      ka::rangeindex rj(j, j+bloc);
      for (size_t i=0; i<N; i += bloc)
      {
        ka::rangeindex ri(i, i+bloc);
        for (size_t k=0; k<K; k += bloc)
        {
          ka::rangeindex rk(k, k+bloc);
//          ka::Spawn<TaskGEMM<T> >( ka::SetArch(ka::ArchCUDA) )
          ka::Spawn<TaskGEMM<T> >(  )
          (  CblasColMajor, CblasNoTrans, CblasNoTrans, (T)1.0,
           B(rk,rj), A(ri,rk), (T)1.0, C(ri,rj) );
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

