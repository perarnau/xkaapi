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
#include <math.h>
#include "matrix.h"
#include "kaapi++" // this is the new C++ interface for Kaapi
#define BLOCSIZE 256


/* Task Error
 * Display the || ||2 matrix norm of A-B
 */
struct TaskError : public ka::Task<2>::Signature<
      ka::R<ka::range2d<double> >, 
      ka::R<ka::range2d<double> > 
> {};

template<>
struct TaskBodyCPU<TaskError> {
  void operator() ( ka::range2d_r<double> A, ka::range2d_r<double> B  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    double error = 0.0;
    for (size_t i=0; i < d0; ++i)
    {
      for (size_t j=0; j < d1; ++j)
      {
        double diff = fabs(A(i,j)-B(i,j));
        error += diff*diff;
      }
    }
    std::cout << "*** Error: " << sqrt(error) << std::endl;
  }
};


/** Compute A*B -> C
*/
struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >,     /* A */
      ka::R<ka::range2d<double> >,     /* B */
      ka::RPWP<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( const ka::StaticSchedInfo* info, 
                   ka::range2d_r<double> A, ka::range2d_r<double> B, 
                   ka::range2d_rpwp<double> C )
  {
    size_t M = A.dim(0);
    size_t K = B.dim(0);
    size_t N = B.dim(1);
    
    /* assume perfect division */
    int sqn = (int)sqrt( (double)info->count_cpu() );
    int bloc_i = M / (2*sqn); // 
    int bloc_j = N / (2*sqn); // /(info->count_cpu()); ///nbloc;

    ka::rangeindex rall(0, K);
    for (size_t j=0; j<N; j += bloc_j)
    {
      ka::rangeindex rj(j, j+bloc_j);
      for (size_t i=0; i<M; i += bloc_i)
      {
        ka::rangeindex ri(i, i+bloc_i);
        ka::Spawn<TaskDGEMM>()(  CblasNoTrans, CblasNoTrans, 1.0, A(ri,rall), B(rall,rj), 1.0, C(ri,rj) );
      }
    }
  }
};




/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    double t0, t1;
    int n     = 8;  // matrix dimension
    int nbloc = 2;  // number of blocs

    if (argc > 1)
        n = atoi(argv[1]);
    if (argc > 2)
        nbloc = atoi(argv[2]);
    
    if ((n / nbloc) ==0) n = nbloc;
    n = (n/nbloc)*nbloc;

    double* dA = (double*) calloc(n* n, sizeof(double));
    double* dB = (double*) calloc(n* n, sizeof(double));
    double* dC = (double*) calloc(n* n, sizeof(double));
    double* dCv = (double*) calloc(n* n, sizeof(double));

   //kaapi_assert_debug( __kaapi_isaligned( dA, 16 ) );
   //kaapi_assert_debug( __kaapi_isaligned( dB, 16 ) );
   //kaapi_assert_debug( __kaapi_isaligned( dC, 16 ) );

    if (0 == dA || 0 == dB || 0 == dC) 
    {
        std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
            << std::endl;
        return;
    }

    // Populate B and C pseudo-randomly - 
    // The matrices are populated with random numbers in the range (-1.0, +1.0)
    for(int i = 0; i < n * n; ++i) {
        dB[i] = (float) ((i * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < n * n; ++i) {
        dA[i] = (float) (((i + 1) * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < n * n; ++i) {
        dC[i] = 0.0;
        dCv[i] = 0.0;
    }

    ka::array<2, double> A(dA, n, n, n);
    ka::array<2, double> B(dB, n, n, n);
    ka::array<2, double> C(dC, n, n, n);
    ka::array<2, double> Cv(dCv, n, n, n);

#if 0
    /* a call to blas to verify */
    t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskSeqMatProduct>() ( A, B, Cv );
    ka::Sync();
    t1 = kaapi_get_elapsedtime();
    std::cout << " Sequential matrix multiply took " << t1-t0 << " seconds." << std::endl;
#endif

std::cout << "-----\n";
#if 1
    // Multiply to get C = A*B 
    t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched(ka::AllCPUType))( A, B, C );
    ka::Sync();
    t1 = kaapi_get_elapsedtime();

    std::cout << " Matrix Multiply " << n << 'x' << n 
              << " #row,#col = " << nbloc << " took " << t1-t0 << " seconds." << std::endl;
#else
    ka::InCache key;
    t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched(ka::AllCPUType, key ))( A, B, C );
    ka::Sync();
    t1 = kaapi_get_elapsedtime();

    std::cout << " Matrix Multiply " << n << 'x' << n 
              << " #row,#col = " << nbloc << " took " << t1-t0 << " seconds." << std::endl;

    t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched(ka::AllCPUType, key ))( A, B, C );
    ka::Sync();
    t1 = kaapi_get_elapsedtime();

    std::cout << " Matrix Multiply " << n << 'x' << n 
              << " #row,#col = " << nbloc << " took " << t1-t0 << " seconds." << std::endl;
#endif

    // If n is small, print the results
    if (n <= 64) 
    {
      ka::Spawn<TaskPrintMatrix<double> >()( std::string("A"), A );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix<double> >()( std::string("B"), B );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix<double> >()( std::string("C"), C );
      ka::Sync();
    } 

    /* a call to blas to verify */
    t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskDGEMM>() (CblasNoTrans, CblasNoTrans, 1.0, A, B, 1.0, Cv );
    ka::Sync();
    t1 = kaapi_get_elapsedtime();
    std::cout << " Sequential matrix multiply took " << t1-t0 << " seconds." << std::endl;

    if (n <= 64) 
    {
      ka::Spawn<TaskPrintMatrix<double> >()( std::string("CSEQ"), Cv );
      ka::Sync();
    }
    ka::Spawn<TaskError>()( Cv, C );
    ka::Sync();

    free(dA);
    free(dB);
    free(dC);
    free(dCv);
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

