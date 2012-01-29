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
#include <cuda.h>
#include "kaapi++" // this is the new C++ interface for Kaapi

#include "../matrix/matrix.h"

#define ALIGN_UP(x) ((double_type*)(((size_t)x+(4096-1))&(~(4096-1)) ))

// missing definition
//extern "C" int kaapi_memory_synchronize(void);

static int BLOCSIZE = 0;

// check results
#define CONFIG_DO_CHECK 1
#if CONFIG_DO_CHECK

# include <stdlib.h>

static int do_check
(const double_type* a, const double_type* b, double_type* c, double_type* c_old, unsigned int n)
{
//  double_type* const tmp = (double_type*) malloc(n * n * sizeof(double_type));
//  if (tmp == NULL) return -1;
  unsigned int i, j, k;
  double_type* const tmp = c_old;

//  for (i = 0; i < n * n; ++i) tmp[i] = 0.;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      for (k = 0; k < n; ++k)
	tmp[i * n +  j] += a[i * n + k] * b[k * n + j];
    }
  }

  int res = -1;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      k = i * n + j;
      if (fabsf(c[k] - tmp[k]) >= 0.001)
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


// fetching task
struct TaskMatFetch : public ka::Task<1>::Signature<
  ka::RW<ka::range2d<double_type> > // C
>{};

template<>
struct TaskBodyCPU<TaskMatFetch> {
  void operator()( ka::range2d_rw<double_type>) {}
};

template<>
struct TaskBodyGPU<TaskMatFetch> {
  void operator()( ka::range2d_rw<double_type>) {}
};

struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double_type> >, /* A */
      ka::R<ka::range2d<double_type> >,  /* B */
      ka::RPWP<ka::range2d<double_type> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( ka::range2d_r<double_type> A, ka::range2d_r<double_type> B, ka::range2d_rpwp<double_type> C )
  {
    size_t M = A.dim(0);
    size_t K = B.dim(0);
    size_t N = B.dim(1);
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
#if 0
	  fprintf(stdout, "TaskMatProduct A([%lu:%lu] [%lu:%lu]) x B([%lu:%lu] [%lu:%lu]) = C([%lu:%lu] [%lu:%lu])\n",
			i, i+bloc, k, k+bloc, k, k+bloc, j, j+bloc, i, i+bloc,
			j, j+bloc
		 );
	  fflush(stdout);
#endif
          ka::Spawn<TaskDGEMM>()(  CblasRowMajor, CblasNoTrans, CblasNoTrans, 1.0, A(ri,rk), B(rk,rj), 1.0, C(ri,rj) );
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
    // av[2] = block_count
    int matrix_size = 512;
    int block_count = 1;
    int verif = 0;

    if( argc > 1 )
	    matrix_size= atoi(argv[1]);
    if( argc > 2 )
	    block_count= atoi(argv[2]);
    if( argc > 3 )
	    verif = 1; // check results ON

    BLOCSIZE = matrix_size / block_count;

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

    ka::Spawn<TaskDLARNV>()( A );
    ka::Spawn<TaskDLARNV>()( B );
    ka::Spawn<TaskDLARNV>()( C );
    ka::Sync();

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

#if 0
    ka::Spawn<TaskPrintMatrix>()( std::string("A"), A );
    ka::Spawn<TaskPrintMatrix>()( std::string("B"), B );
    ka::Sync();
#endif
    // Multiply to get C = A*B 
    double t0 = kaapi_get_elapsedtime();
    //ka::Spawn<TaskMatProduct>()( A, B, C );
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched())( A, B, C );
    ka::Sync();
//    kaapi_memory_synchronize();

    // dont time memory sync for the benchmarks since
    // it does not reflect the execution pipeline
    double t1 = kaapi_get_elapsedtime();
    double tdelta = t1 - t0;

    double gflops = 1.0e-9 * ((2.0 * n * n * n)/(t1-t0));

    fprintf( stdout, "# size bloc threads time GFlop/s\n" );
    fprintf( stdout, "GEMM %d %d %d %.10f %.6f\n", matrix_size, block_count,
		    kaapi_getconcurrency(), tdelta, gflops );
    fflush(stdout);

    //fprintf( stdout, "%d %d %.4f %\n", matrix_size, block_count, tdelta );
    //fprintf( stdout, "Gflop/second %.2f\n", gflops);

    // If n is small, print the results
#if 0
    ka::Spawn<TaskPrintMatrix>()( std::string("C"), C );
    ka::Sync();
#endif

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

