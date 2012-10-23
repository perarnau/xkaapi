/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <cblas.h> 
#include <clapack.h> /* assume MKL/ATLAS clapack version */


#pragma kaapi task value(Order, Uplo, N, lda) readwrite(A{ld=lda; [N][N]})
int clapack_dpotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);


#pragma kaapi task value(Order, Side, Uplo, TransA, Diag, M, N, alpha, lda, ldb) \
                   read(A{ld=lda; [N][N]}) \
                   readwrite(B{ld=ldb; [N][N]})
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb);

#pragma kaapi task value(Order, Uplo, Trans, N, K, alpha, lda, beta, ldc) \
                   read(A{ld=lda; [N][N]}) \
                   readwrite(C{ld=ldc; [N][N]})
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc);

#pragma kaapi task value(Order, TransA, TransB, M, N, K, alpha, lda, ldb, beta, ldc) \
                   read(A{ld=lda; [M][K]}, B{ld=ldb; [K][N]}) \
                   readwrite(C{ld=ldc; [M][N]})
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);


/* Generate a random matrix symetric definite positive matrix of size m x m 
   - it will be also interesting to generate symetric diagonally dominant 
   matrices which are known to be definite postive.
*/
static void generate_matrix(double* A, size_t m)
{
  // 
  for (size_t i = 0; i< m; ++i)
  {
    for (size_t j = 0; j< m; ++j)
      A[i*m+j] = 1.0 / (1.0+i+j);
    A[i*m+i] = m*1.0; 
  }
}


/* Block Cholesky factorization A <- L * L^t
   Lower triangular matrix, with the diagonal, stores the Cholesky factor.
*/
void Cholesky( double* A, int N, size_t blocsize )
{
#pragma kaapi parallel scheduling(static)
  for (size_t k=0; k < N; k += blocsize)
  {
    clapack_dpotrf(
      CblasRowMajor, CblasLower, blocsize, &A[k*N+k], N
    );

    for (size_t m=k+blocsize; m < N; m += blocsize)
    {
      cblas_dtrsm
      (
        CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
        blocsize, blocsize, 1., &A[k*N+k], N, &A[m*N+k], N
      );
    }

    for (size_t m=k+blocsize; m < N; m += blocsize)
    {
      cblas_dsyrk
      (
        CblasRowMajor, CblasLower, CblasNoTrans,
        blocsize, blocsize, -1.0, &A[m*N+k], N, 1.0, &A[m*N+m], N
      );
      for (size_t n=k+blocsize; n < m; n += blocsize)
      {
        cblas_dgemm
        (
          CblasRowMajor, CblasNoTrans, CblasTrans,
          blocsize, blocsize, blocsize, -1.0, &A[m*N+k], N, &A[n*N+k], N, 1.0, &A[m*N+n], N
        );
      }
    }
  }
}


/* Do one run for cholesky
*/
void doone_exp( int N, int block_count )
{
  size_t blocsize = N / block_count;

  printf("N         : %i\n", N);
  printf("size block: %i\n", blocsize);
  printf("#blocks   : %i\n", block_count);
  double* A = 0;
  if (0 != posix_memalign((void**)&A, 4096, N*N*sizeof(double)))
  {
    printf("Fatal Error. Cannot allocate matrice A, errno: %i\n", errno);
    return;
  }

  generate_matrix(A, N);

  Cholesky(A, N, blocsize);
  
  free(A);
}



/* main entry point
*/
int main(int argc, char** argv)
{
  // matrix dimension
  int n = 32;
  if (argc > 1)
    n = atoi(argv[1]);

  // block count
  int block_count = 2;
  if (argc > 2)
    block_count = atoi(argv[2]);
    
  doone_exp( n, block_count );

  return 0;
}
