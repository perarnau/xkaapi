/*
** kaapi_impl.h
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
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
#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <cblas.h>
//#include <clapack.h>
#include <math.h>
#include <sys/types.h>
#include "kaapi++" // this is the new C++ interface for Kaapi


/* Generate a random matrix of size m x m */
static void generate_matrix(double* c, size_t m)
{
  // generate a random matrix
  const size_t mm = m * m;

  // generate a down triangular matrix
  for (size_t i = 0; i < mm; ++i)
    c[i] = drand48() * 1000.f;
}


/* Task Print Matrix
 * this task prints the matrix using Maple matrix format
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


/* Task Print Matrix LU
 * assume that the matrix stores an LU decomposition with L lower triangular matrix with unit diagonal
   and U upper triangular matrix, then print both L and U using the Maple matrix format.
 */
struct TaskPrintMatrixLU : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double> > > {};

template<>
struct TaskBodyCPU<TaskPrintMatrixLU> {
  void operator() ( std::string msg, ka::range2d_r<double> A  )
  {
    size_t d0 = A.dim(0);
    size_t d1 = A.dim(1);
    std::cout << "U :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j<i; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << 0 << (j == d1-1 ? "" : ", ");
      }
      for (size_t j=i; j < d1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;

    std::cout << "L :=matrix( [" << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      std::cout << "[";
      for (size_t j=0; j < i; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific 
                  << A(i,j) << (j == d1-1 ? "" : ", ");
      }
      std::cout << std::setw(18) 
                << std::setprecision(15) 
                << std::scientific 
                << 1 << (i == d1-1 ? "" : ", ");
      for (size_t j=i+1; j<d1; ++j)
      {
        std::cout << std::setw(18) 
                  << std::setprecision(15) 
                  << std::scientific << 0 << (j == d1-1 ? "" : ", ");
      }
      std::cout << "]" << (i == d0-1 ? ' ' : ',') << std::endl;
    }
    std::cout << "]);" << std::endl;
    std::cout << "evalm( L &* U  - A);" << std::endl;
  }
};



/* Compute the norm || A - L*U ||infinity
 * The norm value serves to detec an error in the computation
 */
struct TaskNormMatrix: public ka::Task<3>::Signature<
  ka::W<double>, /* norm */
  ka::R<ka::range2d<double> >, /* A */
  ka::R<ka::range2d<double> > /* LU */
>{};
template<>
struct TaskBodyCPU<TaskNormMatrix> {
  void operator() ( ka::pointer_w<double> norm, ka::range2d_r<double> A, ka::range2d_r<double> LU )
  {
    int k;
    const double* dA = A.ptr();
    int lda_A = A.lda();
    const double* dLU = LU.ptr();
    int lda_LU = LU.lda();
    int M = A.dim(0);
    int N = A.dim(1);
    *norm = 0.0;

    double max = 0.0;
    /* makes the triangular - triangluar matrix product by the 'hand' */
    for (int i=0; i<M; ++i)
    {
      for (int j=0; j<N; ++j)
      {
        const double* Lik = dLU+i*lda_LU;
        const double* Ukj = dLU+j;
        double c=0.0;
        int kstop = (i<j ? i : j);
        for (k=0; k < kstop; ++k, ++Lik, Ukj += lda_LU)
          c += Lik[0] * Ukj[0];
        if (i == kstop) 
        { 
          /* L(i,i) = 1.0 */
          c += *Ukj;
        }
        else
          c += *Lik * *Ukj;

        double error_ij = fabs(dA[i*lda_A+j] - c);
        if (error_ij > max) 
          max = error_ij;
      }
    }
    *norm = max;
  }
};



/* Compute inplace LU factorization of A, ie L and U such that A = L * U
   with L lower triangular, with unit diagonal and U upper triangular.
*/
struct TaskDPOTRF: public ka::Task<1>::Signature<
  ka::RW<ka::range2d<double> > /* A */
>{};
template<>
struct TaskBodyCPU<TaskDPOTRF> {
  void operator()
  ( ka::range2d_rw<double> A )
  {
    // dpotrf(uplo, n, a, lda, info)
    // compute the cholesky factorization
    // A = U**T * U,  if UPLO = 'U', or
    // A = L  * L**T,  if UPLO = 'L'

    int N = A.dim(0);
    int M = A.dim(1);
    int k_stop = std::min(N, M);

#if 0
    /* Simple code: correct ||L*U -A || very small */
    for ( int k = 0; k < k_stop; ++k )
    {
      double pivot = 1.0 / A(k,k);
      for (int i = k+1 ; i < N ; ++i )
        A(i,k) *= pivot;
      for (int i= k+1 ; i < N; ++i)
        for (int j= k+1 ; j < M; ++j)
          A(i,j) -=  A(i,k)*A(k,j);
    }
#else
    /* More optimized code: correct ||L*U -A || very small */
    double* dk = A.ptr(); /* &A(k,0) */
    int lda = A.lda();
    
    for ( int k = 0; k < k_stop; ++k, dk += lda )
    {
      double pivot = 1.0 / dk[k];

      /* column normalization */
      double* dik = dk+k+lda; /* A(i,k) i=k+1 */
      for (int i = k+1 ; i < N ; ++i, dik += lda )
        *dik *= pivot;

      /* submatrix update */
      dik = dk+k+lda; /* A(i,k) i=k+1 */
      for (int i= k+1 ; i < N; ++i, dik += lda)
      {
        double* dij = dik+1;
        double* dkj = dk+k+1;
        for (int j= k+1 ; j < M; ++j, ++dij,++dkj)
          *dij -=  *dik * *dkj;
      } 
    }
#endif
  }
};



/* Solve : L(rk,rk) * X =  * A(rk,rj) 
    ie:  X <- L(rk,rk)^-1 * A(rk,rj) 
*/
struct TaskDTRSM_left: public ka::Task<2>::Signature<
      ka::R<ka::range2d<double> >, /* Akk */
      ka::RW<ka::range2d<double> > /* Akj */
>{};
template<>
struct TaskBodyCPU<TaskDTRSM_left> {
  void operator()( ka::range2d_r<double> Akk, ka::range2d_rw<double> Akj )
  {
    // dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
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

    const double* const a = Akk.ptr();
    const int lda = Akk.lda();

    double* const b = Akj.ptr();
    const int ldb   = Akj.lda();
    const int n     = Akj.dim(0);
    const int m     = Akj.dim(1);

    cblas_dtrsm
    (
     CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
     n, m, 1., a, lda, b, ldb
    );
  }
};

/* Solve : X * U(rk,rk) =  A(ri,rk) 
    ie:  X <- A(ri,rk) * U(rk,rk)^-1

*/
struct TaskDTRSM_right: public ka::Task<2>::Signature<
      ka::R<ka::range2d<double> >, /* Akk */
      ka::RW<ka::range2d<double> > /* Aik */
>{};
template<>
struct TaskBodyCPU<TaskDTRSM_right> {
  void operator()( ka::range2d_r<double> Akk, ka::range2d_rw<double> Aik )
  {
    const double* const a = Akk.ptr();
    const int lda = Akk.lda();

    double* const b = Aik.ptr();
    const int ldb = Aik.lda();
    const int n = Aik.dim(0); // b.rows();
    const int m = Aik.dim(1); // b.cols();

    cblas_dtrsm
    (
      CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
      n, m, 1., a, lda, b, ldb
    );
  }
};


/* DGEMM rountine to compute
    Aij <- Aij - Aik * Akj
*/
struct TaskDGEMM: public ka::Task<3>::Signature<
      ka::R<ka::range2d<double> >, /* Aik*/
      ka::R<ka::range2d<double> >, /* Akj */
      ka::RW<ka::range2d<double> > /* Aij */
>{};
template<>
struct TaskBodyCPU<TaskDGEMM> {
  void operator()
  (
   ka::range2d_r<double> Aik,
   ka::range2d_r<double> Akj,
   ka::range2d_rw<double> Aij
  )
  {
    // dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    //
    // C = alpha * op(A) * op(B) + beta * C

    const double* const a = Aik.ptr();
    const double* const b = Akj.ptr();
    double* const c = Aij.ptr();

    const int m = Aik.dim(0); 
    const int n = Aik.dim(1); // eq. to Akj.rows();
    const int k = Akj.dim(1); 

    const int lda = Aik.lda();
    const int ldb = Akj.lda();
    const int ldc = Aij.lda();

    cblas_dgemm
    (
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, -1., a, lda, b, ldb, 1., c, ldc
    );
  }
};


/* Block LU factorization
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
      // A(rk,rk) = L(rk,rk) * U(rk,rk) <- LU( A(rk,rk) 
      ka::Spawn<TaskDPOTRF>()( A(rk,rk) );

      for (size_t j=k+blocsize; j<N; j += blocsize)
      {
        ka::rangeindex rj(j, j+blocsize);
        // A(rk,rj) <- L(rk,rk)^-1 * A(rk,rj) 
        ka::Spawn<TaskDTRSM_left>()(A(rk,rk), A(rk,rj));
      }
      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        // A(ri,rk) <- A(ri,rk) * U(rk,rk)^-1
        ka::Spawn<TaskDTRSM_right>()(A(rk,rk), A(ri,rk));
      }

      for (size_t i=k+blocsize; i<N; i += blocsize)
      {
        ka::rangeindex ri(i, i+blocsize);
        for (size_t j=k+blocsize; j<N; j += blocsize)
        {
          ka::rangeindex rj(j, j+blocsize);
          // A(ri,rj) <- A(ri,rj) - A(ri,rk)*A(rk,rj)
          ka::Spawn<TaskDGEMM>()(A(ri,rk), A(rk,rj), A(ri,rj));
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
    // matrix dimension
    int n = 32;
    if (argc > 1)
      n = atoi(argv[1]);

    // block count
    int block_count = 2;
    if (argc > 2)
      block_count = atoi(argv[2]);
      
    // Make verification ?
    int verif = 0;
    if (argc >3)
      verif = atoi(argv[3]);

    const int block_size = n / block_count;
    n = block_count * block_size;

    double* dA = (double*) calloc(n* n, sizeof(double));
    double* dAcopy = (double*) calloc(n* n, sizeof(double));
    if ((0 == dA) || (dAcopy ==0))
    {
      std::cout << "Fatal Error. Cannot allocate matrices A, "
                << std::endl;
      return;
    }

    generate_matrix(dA, n);
    /* copy the matrix to compute the norm */
    memcpy(dAcopy, dA, n*n*sizeof(double) );

    ka::array<2,double> A(dA, n, n, n);

#if 0
    if (n <= 32) {
      /* output respect the Maple format */
      ka::Spawn<TaskPrintMatrix>()("A", A);
      ka::Sync();
    }
#endif

    std::cout << "Start LU with " << block_count << 'x' << block_count << " blocs of matrix of size " << n << 'x' << n << std::endl;
    // LU factorization of A using ka::
    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<TaskLU>(ka::SetStaticSched())(block_size, A);
    ka::Sync();
    double t1 = kaapi_get_elapsedtime();

    std::cout << " LU took " << t1-t0 << " seconds." << std::endl;

    if (verif)
    {
#if 0
      /* If n is small, print the results */
      if (n <= 32) 
      {
        ka::Spawn<TaskPrintMatrixLU>()( std::string(""), A );
        ka::Sync();
      }
      else  /* else compute the norm || A - L*U ||inf */
#endif
      {
        double norm;
        ka::Spawn<TaskNormMatrix>()
	  ( &norm, ka::array<2,double>(dAcopy, n, n, n), A );
        ka::Sync();
        std::cout << "Error ||A-LU||inf :" << norm << std::endl;
      }
    }

    free(dA);
    free(dAcopy);
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
