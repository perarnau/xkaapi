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
#include <string>
#include "kaapi++" // this is the new C++ interface for Kaapi
#include <cblas.h>

#define THRESHOLD 2

struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* A */
      ka::R<ka::range2d<double> >,  /* B */
      ka::W<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( ka::range2d_r<double> A, ka::range2d_r<double> B, ka::range2d_w<double> C )
  {
    size_t M = A.dim(0);
    size_t N = B.dim(0);
    size_t K = C.dim(1);
    
    /* a call to blas should be more performant here */
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0, 
        A.ptr(), A.lda(),
        B.ptr(), B.lda(),
        1.0, 
        C.ptr(), C.lda()
    );

  }
};


/* Task Print
 * this task prints the sum of the entries of an array 
 * each entries is view as a pointer object:
    array<1,R<int> > means that each entry may be read by the task
 */
struct TaskPrintMatrix : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double> > > {};

template<>
struct TaskBodyCPU<TaskPrintMatrix> {
  void operator() ( std::string msg, ka::range2d_r<double> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    std::cout << msg << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      for (size_t j=0; j < d1; ++j)
        std::cout << A(i,j) << " ";
      std::cout << std::endl;
    }
  }
};



/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    int n = 2*THRESHOLD;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    double* dA = (double*) calloc(n* n, sizeof(double));
    double* dB = (double*) calloc(n* n, sizeof(double));
    double* dC = (double*) calloc(n* n, sizeof(double));
    if (0 == dA || 0 == dB || 0 == dC) {
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
        dC[i] = (float) (((i + 1) * i) % 1024 - 512) / 512;
    }

    ka::array<2,double> A(dA, n, n, n);
    ka::array<2,double> B(dB, n, n, n);
    ka::array<2,double> C(dC, n, n, n);
    // Multiply to get A = B*C 
    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>()( A, B, C );
    ka::Sync();
    double t1 = kaapi_get_elapsedtime();

    std::cout << " Matrix Multiply took " << t1-t0 << " seconds." << std::endl;

    // If n is small, print the results
    if (n <= 16) {
      ka::Spawn<TaskPrintMatrix>()( std::string("A="), A );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix>()( std::string("B="), B );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix>()( std::string("C="), C );
      ka::Sync();
    }

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

