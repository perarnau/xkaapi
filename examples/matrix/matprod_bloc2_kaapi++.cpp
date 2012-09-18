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

static int BLOCSIZE = 0;

/** Parallel bloc matrix product (second level)
*/
struct TaskMatProduct2: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* A */
      ka::R<ka::range2d<double> >,  /* B */
      ka::RPWP<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct2> {
  void operator()( 
    ka::range2d_r<double> A, 
    ka::range2d_r<double> B, 
    ka::range2d_rpwp<double> C 
  )
  {
    size_t M = A->dim(0);
    size_t K = B->dim(0);
    size_t N = B->dim(1);
    int bloc = BLOCSIZE/4;
    
    for (size_t i=0; i<M; i += bloc)
    {
      ka::rangeindex ri(i, i+bloc);
      for (size_t j=0; j<N; j += bloc)
      {
        ka::rangeindex rj(j, j+bloc);
        for (size_t k=0; k<K; k += bloc)
        {
          ka::rangeindex rk(k, k+bloc);
          ka::Spawn<TaskDGEMM>()(  CblasNoTrans, CblasNoTrans, 1.0, A(ri,rk), B(rk,rj), 1.0, C(ri,rj) );
        }
      }
    }
  }
};

/** Parallel bloc matrix product (upper level)
*/
struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* A */
      ka::R<ka::range2d<double> >,  /* B */
      ka::RPWP<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( 
    ka::range2d_r<double> A, 
    ka::range2d_r<double> B, 
    ka::range2d_rpwp<double> C 
  )
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
          ka::Spawn<TaskMatProduct2>(ka::SetStaticSched())(  A(ri,rk), B(rk,rj), C(ri,rj) );
        }
      }
    }
  }
};

/*
*/
static void Usage(const char* progname)
{
  std::cerr << "**** usage: " << progname << " N NB [verif]\n"
            << " N    : matrix dimension\n"
            << " BS   : block size\n"
            << " verif: optional value 1 do verification, 0 do not verify, default 0"
            << std::endl;
  exit(1);
}

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    int n = 2;
    int nb = 1;
    int verif = 0;
    
    if (argc <3) 
      Usage(argv[0]);

    if (argc > 1) 
      n = atoi(argv[1]);

    if (argc > 2) 
      BLOCSIZE = atoi(argv[2]);

    if (argc >3)
      verif = atoi(argv[3]);

    nb = n / BLOCSIZE;
    n = BLOCSIZE * nb;

    double* dA = (double*) calloc(n* n, sizeof(double));
    double* dB = (double*) calloc(n* n, sizeof(double));
    double* dC = (double*) calloc(n* n, sizeof(double));
    if (0 == dA || 0 == dB || 0 == dC) 
    {
        std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
            << std::endl;
        return;
    }

    // Populate B and C pseudo-randomly - 
    // The matrices are populated with random numbers in the range (-1.0, +1.0)
    for(int i = 0; i < n * n; ++i) {
        dB[i] = 2.0*drand48()-1.0;
    }
    for(int i = 0; i < n * n; ++i) {
        dA[i] = 2.0*drand48()-1.0;
    }
    for(int i = 0; i < n * n; ++i) {
        dC[i] = 0.0;
    }

    ka::array<2, double> A(dA, n, n, n);
    ka::array<2, double> B(dB, n, n, n);
    ka::array<2, double> C(dC, n, n, n);

    // Multiply to get C = A*B 
    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskMatProduct>(ka::SetStaticSched())( A, B, C );
    ka::Sync();
    double t1 = kaapi_get_elapsedtime();

    std::cout << " Matrix Multiply " << n << 'x' << n << " took " << t1-t0 << " seconds." << std::endl;

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
    
    if (verif)
    {
      /* a call to blas to verify */
      double* dCv = (double*)calloc(n* n, sizeof(double));
      ka::array<2, double> Cv(dCv, n, n, n);

      t0 = kaapi_get_elapsedtime();
      ka::Spawn<TaskDGEMM>(ka::SetStaticSched()) (CblasNoTrans, CblasNoTrans, 1.0, A, B, 1.0, Cv );
      ka::Sync();
      t1 = kaapi_get_elapsedtime();

      std::cout << " Sequential matrix multiply took " << t1-t0 << " seconds." << std::endl;

      double error;
      ka::Spawn<TaskNorm2>()( &error, Cv, C );
      ka::Sync();
      std::cout << "*** Error: " << error << std::endl;
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

