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
#include "kaapi++" // this is the new C++ interface for Kaapi

#define USE_CBLAS 1
#if defined(USE_CBLAS)
#include <cblas.h>
#endif

#define BLOCSIZE 256

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
    std::cout << msg << " :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j < d1; ++j)
      {
        std::cout << std::setw(18) << std::setprecision(15) << std::scientific << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;
  }
};


/* Task Error
 * Compute the || ||2 matrix norm of A-B
 */
struct TaskError : public ka::Task<2>::Signature<ka::R<ka::range2d<double> >, ka::R<ka::range2d<double> > > {};

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


/**
*/
struct TaskSeqMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* A */
      ka::R<ka::range2d<double> >,  /* B */
      ka::RW<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskSeqMatProduct> {
  void operator()( ka::range2d_r<double> A, ka::range2d_r<double> B, ka::range2d_rw<double> C )
  {
    size_t M = A.dim(0);
    size_t K = B.dim(0);
    size_t N = B.dim(1);
#if 0
    std::cout << kaapi_get_self_kid() << "::In TaskSeqMul A:" << A.ptr() << " dim: " << M << 'x' << K 
              << ", B:" << B.ptr() << " dim: " << K << 'x' << N
              << ", C:" << C.ptr()
              << std::endl;
#endif
              
#if defined(USE_CBLAS)
    /* a call to blas should be more performant here */
    cblas_dgemm(
        CblasRowMajor, 
        CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0, 
        A.ptr(), A.lda(),
        B.ptr(), B.lda(),
        1.0, 
        C.ptr(), C.lda()
    );
#else
    for (size_t i =0; i<N;++i)
      for (size_t j =0; j<M; ++j)
        for (size_t k =0; k<K; ++k)
          C(i,j) += A(i,k)*B(k,j);
#endif
  }
};


struct TaskMatProduct: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* A */
      ka::R<ka::range2d<double> >,  /* B */
      ka::RPWP<ka::range2d<double> >   /* C */
>{};

template<>
struct TaskBodyCPU<TaskMatProduct> {
  void operator()( ka::range2d_r<double> A, ka::range2d_r<double> B, ka::range2d_rpwp<double> C )
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
          ka::Spawn<TaskSeqMatProduct>()( A(ri,rk), B(rk,rj), C(ri,rj) );
        }
      }
    }
  }
};




/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    int n = 2;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    n *= BLOCSIZE;

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
        dB[i] = (float) ((i * i) % 1024 - 512) / 512;
    }
    for(int i = 0; i < n * n; ++i) {
        dA[i] = (float) (((i + 1) * i) % 1024 - 512) / 512;
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
      ka::Spawn<TaskPrintMatrix>()( std::string("A"), A );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix>()( std::string("B"), B );
      ka::Sync();
      ka::Spawn<TaskPrintMatrix>()( std::string("C"), C );
      ka::Sync();
    } else {
      /* a call to blas to verify */
      double* dCv = (double*) calloc(n* n, sizeof(double));
      double t0 = kaapi_get_elapsedtime();
#if defined(USE_CBLAS)
      cblas_dgemm(
          CblasRowMajor, 
          CblasNoTrans, CblasNoTrans,
          n, n, n, 1.0, 
          dA, n,
          dB, n,
          1.0, 
          dCv, n
      );
#endif
      double t1 = kaapi_get_elapsedtime();
      std::cout << " Sequential matrix multiply took " << t1-t0 << " seconds." << std::endl;

      ka::array<2, double> Cv(dCv, n, n, n);
      ka::Spawn<TaskError>()( Cv, C );
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

