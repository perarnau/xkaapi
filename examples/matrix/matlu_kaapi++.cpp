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


#if CONFIG_USE_GSL

#include <sys/types.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

static void do_gsl_lufact(gsl_matrix* a)
{
  gsl_permutation* const p = gsl_permutation_alloc(a->size1);
  int s;
  gsl_linalg_LU_decomp(a, p, &s);
  gsl_permutation_free(p);
}

static gsl_matrix* allocate_gsl_matrix(size_t w)
{
  gsl_matrix* const m = gsl_matrix_alloc(w, w);
  return m;
}

static void free_gsl_matrix(gsl_matrix* m)
{
  gsl_matrix_free(m);
}

__attribute__((unused))
static gsl_matrix* create_gsl_matrix(ka::array<2, double> array)
{
  gsl_matrix* const m = allocate_gsl_matrix(array.dim(0));

  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j)
      gsl_matrix_set(m, i, j, array(i, j));

  return m;
}

__attribute__((unused))
static gsl_matrix* create_gsl_matrix
(const double* array, size_t w)
{
  gsl_matrix* const m = allocate_gsl_matrix(w);

  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j)
      gsl_matrix_set(m, i, j, array[i * w + j]);

  return m;
}

__attribute__((unused))
static void fill_gsl_matrix(gsl_matrix* m)
{
  for (size_t i = 0; i < m->size1; ++i)
    for (size_t j = 0; j < m->size2; ++j)
      gsl_matrix_set(m, i, j, 1.);
}

static void print_gsl_matrix(const gsl_matrix* m)
{
  for (size_t i = 0; i < m->size1; ++i)
  {
    for (size_t j = 0; j < m->size2; ++j)
      printf("%lf ", gsl_matrix_get(m, i, j));
    printf("\n");
  }
  printf("\n");
}

#endif // CONFIG_USE_GSL


struct TaskDPOTRF: public ka::Task<1>::Signature<
      ka::RW<ka::range2d<double> > /* A */
>{};
template<>
struct TaskBodyCPU<TaskDPOTRF> {
  void operator()( ka::range2d_rw<double> A )
  {
  }
};

struct TaskDTRSM: public ka::Task<2>::Signature<
      ka::R<ka::range2d<double> >, /* Akk */
      ka::RW<ka::range2d<double> > /* Amk */
>{};
template<>
struct TaskBodyCPU<TaskDTRSM> {
  void operator()( ka::range2d_r<double> Akk, ka::range2d_rw<double> Amk )
  {
    // dtrsm(side, uplo, transa, diag, m, n, alpha, lda, ldb)
    //
    // solve the following eq.
    // op(A) * X = alpha * B (0) or
    // X * op(A) = alpha * B (1)
    //
    // side: 'l' or 'r' for (0) or (1)
    // uplo: 'u' or 'l' for upper or lower matrix
    // transa: 'n' or 't' for transposed or not
    // diag: 'u' or 'n' if a is assumed to be unit triangular or not
    //
    // the matrix X is overwritten on B
  }
};

struct TaskDSYRK: public ka::Task<2>::Signature<
      ka::R<ka::range2d<double> >, /* Ann */
      ka::RW<ka::range2d<double> > /* Amk */
>{};
template<>
struct TaskBodyCPU<TaskDSYRK> {
  void operator()( ka::range2d_r<double> Akk, ka::range2d_rw<double> Amk )
  {
    // dsyrk(uplo, tran, n, k, alpha, lda, beta, c, ldc)
    //
    // C = alpha * A * A' + beta * C
  }
};

struct TaskDGEMM: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* Amk*/
      ka::R<ka::range2d<double> >, /* Ank */
      ka::RW<ka::range2d<double> > /* Amn */
>{};
template<>
struct TaskBodyCPU<TaskDGEMM> {
  void operator()( ka::range2d_r<double> Amk, ka::range2d_r<double> Ank, ka::range2d_rw<double> Amn )
  {
    // dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    //
    // C = alpha * op(A) * op(B) + beta * C
  }
};


/* Simple algorithm:
    FOR k = 0..TILES−1 
      A[k][k] <− DPOTRF(A[k][k]) 
      FOR m = k+1..TILES−1
        A[m][k] <− DTRSM(A[k][k] , A[m][k]) 
      FOR n = k+1..TILES−1
        A[n][n] <− DSYRK(A[n][k] , A[n][n]) 
        FOR m = n+1..TILES−1
          A[m][n] <− DGEMM(A[m][k] , A[n][k] , A[m][n])    
*/

struct TaskLU: public ka::Task<2>::Signature<
      int,
      ka::RPWP<ka::range2d<double> > /* A */
>{};

template<>
struct TaskBodyCPU<TaskLU> {
  void operator()( int blocsize, ka::range2d_rpwp<double> A )
  {
    size_t N = A.dim(0);
    for (size_t k=0; k<N; k += blocsize)
    {
      ka::rangeindex rk(k, k+blocsize);
      ka::Spawn<TaskDPOTRF>()( A(rk,rk) );

      for (size_t m=k+blocsize; m<N; m += blocsize)
      {
        ka::rangeindex rm(m, m+blocsize);
        ka::Spawn<TaskDTRSM>()(A(rk,rk), A(rm,rk));
      }

      for (size_t n=k+blocsize; n<N; n += blocsize)
      {
        ka::rangeindex rn(n, n+blocsize);
        ka::Spawn<TaskDSYRK>()(A(rn,rk), A(rn,rn));
        for (size_t m=n+blocsize; m<N; m += blocsize)
        {
          ka::rangeindex rm(m, m+blocsize);
          ka::Spawn<TaskDGEMM>()(A(rm,rk), A(rn,rk), A(rm,rn));
        }
      }
    }
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
    int n = 8;
    int nbloc = 2;
    int blocsize = n / nbloc;

    if (argc > 1) {
        nbloc = atoi(argv[1]);
    }
    if (argc > 2) {
        blocsize = atoi(argv[2]);
        n = nbloc*blocsize;
    }
    n = ((n+nbloc-1)/nbloc)*nbloc;
    blocsize = n/nbloc;

    double* dA = (double*) calloc(n* n, sizeof(double));
    if (0 == dA) 
    {
      std::cout << "Fatal Error. Cannot allocate matrices A, B, and C."
          << std::endl;
      return;
    }

    // Populate B and C pseudo-randomly - 
    // The matrices are populated with random numbers in the range (0.0, +1.0)
    for(int i = 0; i < n * n; ++i) {
        dA[i] = (float) (((i + 1) * i) % 1024) / 1024;
    }

    ka::array<2,double> A(dA, n, n, n);

#if CONFIG_USE_GSL
    gsl_matrix* gslA = create_gsl_matrix(A);
#endif

    std::cout << "Start LU with " << nbloc << 'x' << nbloc << " blocs of matrix of size " << n << 'x' << n << std::endl;

    // LU factorization of A using ka::
    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskLU>(ka::SetStaticSched())( blocsize, A );
    ka::Sync();
    double t1 = kaapi_get_elapsedtime();

#if CONFIG_USE_GSL
    // LU factorization of A using gsl
    do_gsl_lufact(gslA);
#endif

    std::cout << " LU took " << t1-t0 << " seconds." << std::endl;

    // If n is small, print the results
    if (n <= 16) {
      ka::Spawn<TaskPrintMatrix>()( std::string("A="), A );
      ka::Sync();

#if CONFIG_USE_GSL
      printf("--\n");
      print_gsl_matrix(gslA);
#endif
    }

#if CONFIG_USE_GSL
    free_gsl_matrix(gslA);
#endif

    free(dA);
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

