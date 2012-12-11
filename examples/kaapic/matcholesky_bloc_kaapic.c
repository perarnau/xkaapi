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
#include <math.h>
#include <sys/types.h>
#include <cblas.h>   // 
#include <clapack.h> // assume MKL/ATLAS clapack version
#include "kaapic.h" // this is the C interface for Kaapi

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


/*
*/
#define DIM 32
static void convert_to_blocks(long NB, long N, double *Alin, double* A[DIM][DIM] )
{
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      A[j/NB][i/NB][(i%NB)*NB+j%NB] = Alin[i*N+j];
    }
  }

}

/* Block Cholesky factorization A <- L * L^t
   Lower triangular matrix, with the diagonal, stores the Cholesky factor.
*/
void TaskCholesky( int N, int NB, double* ptr )
{
  //kaapic_begin_parallel(KAAPIC_FLAG_STATIC_SCHED);
  typedef double* TYPE[DIM][DIM]; 
  TYPE* A = (TYPE*)ptr;

  for (int k=0; k < N; ++k)
  {
    kaapic_spawn( (void*)0, 5, clapack_dpotrf,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasRowMajor,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasLower,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
        KAAPIC_MODE_RW, KAAPIC_TYPE_DBL, NB*NB, (*A)[k][k], 
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB
    );

    for (int m=k+1; m < N; ++m)
    {
      kaapic_spawn( (void*)0, 12, cblas_dtrsm,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasRowMajor,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasRight,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasLower,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasTrans,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasNonUnit,
          KAAPIC_MODE_V, KAAPIC_TYPE_DBL, 1, (double)1.0,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_R, KAAPIC_TYPE_DBL, NB*NB, (*A)[k][k],
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_RW, KAAPIC_TYPE_DBL, NB*NB, (*A)[m][k], 
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB
      );
    }

    for (int m=k+1; m < N; ++m)
    {
      kaapic_spawn( (void*)0, 11, cblas_dsyrk,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasRowMajor,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasLower, 
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasNoTrans,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_V, KAAPIC_TYPE_DBL, 1, -1.0,
          KAAPIC_MODE_R, KAAPIC_TYPE_DBL, NB*NB, (*A)[m][k],
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
          KAAPIC_MODE_V, KAAPIC_TYPE_DBL, 1, 1.0,
          KAAPIC_MODE_RW, KAAPIC_TYPE_DBL, NB*NB, (*A)[m][m],
          KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB
      );
      for (int n=k+1; n < m; ++n)
      {
        kaapic_spawn( (void*)0, 14, cblas_dgemm,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasRowMajor,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasNoTrans,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)CblasTrans,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
            KAAPIC_MODE_V, KAAPIC_TYPE_DBL, 1, -1.0,
            KAAPIC_MODE_R, KAAPIC_TYPE_DBL, NB*NB, (*A)[m][k],
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
            KAAPIC_MODE_R, KAAPIC_TYPE_DBL, NB*NB, (*A)[n][k],
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB,
            KAAPIC_MODE_V, KAAPIC_TYPE_DBL, 1, 1.0,
            KAAPIC_MODE_RW, KAAPIC_TYPE_DBL, NB*NB, (*A)[m][n],
            KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, NB
        );
      }
    }
  }
  //kaapic_end_parallel(KAAPIC_FLAG_STATIC_SCHED);
}



/* Main of the program
*/
void doone_exp( int n, int block_count, int niter, int verif )
{
  int NB = n / block_count;
  
  //n = block_count * global_blocsize;

  double t0, t1;
  double* dA = (double*) calloc(n* n, sizeof(double));
  if (0 == dA)
  {
    fprintf(stderr, "Fatal Error. Cannot allocate matrice A\n ");
    return;
  }

  // blocked matrix
  double* dAbloc[DIM][DIM];
  for (long i = 0; i < DIM; i++)
    for (long j = 0; j < DIM; j++)
    {
     dAbloc[i][j] = 0;
     posix_memalign( (void**)&(dAbloc[i][j]), 128, NB*NB*sizeof(double));
    }

            
  // Cholesky factorization of A 
  double sumt = 0.0;
  double sumgf = 0.0;
  double sumgf2 = 0.0;
  double sd;
  double sumt_exec = 0.0;
  double sumgf_exec = 0.0;
  double sumgf2_exec = 0.0;
  double sd_exec;
  double gflops;
  double gflops_max = 0.0;
  double gflops_exec;
  double gflops_exec_max = 0.0;

  /* formula used by plasma in time_dpotrf.c */
  double fp_per_mul = 1;
  double fp_per_add = 1;
  double fmuls = (n * (1.0 / 6.0 * n + 0.5 ) * n);
  double fadds = (n * (1.0 / 6.0 * n ) * n);
      
  printf("%6d %5d %5d ", (int)n, (int)kaapic_get_concurrency(), (int)(n/NB) );

  generate_matrix( dA, n );
  convert_to_blocks(NB, n, dA, dAbloc);
  for (int i=0; i<niter; ++i)
  {
    t0 = kaapic_get_time();
    kaapic_spawn( 0, 3, TaskCholesky,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)block_count,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, (int)NB, 
        KAAPIC_MODE_V, KAAPIC_TYPE_PTR, 1, (uintptr_t)&dAbloc
    );
    kaapic_sync();
    t1 = kaapic_get_time();

    gflops = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / (t1-t0);
    if (gflops > gflops_max) gflops_max = gflops;
//printf("Gflops= %9.3f\n", gflops );

    sumt += (double)(t1-t0);
    sumgf += gflops;
    sumgf2 += gflops*gflops;

    gflops_exec = 1e-9 * (fmuls * fp_per_mul + fadds * fp_per_add) / 1.0;
    if (gflops_exec > gflops_exec_max) gflops_exec_max = gflops_exec;
    
    sumt_exec += (double)(1.0);
    sumgf_exec += gflops_exec;
    sumgf2_exec += gflops_exec*gflops_exec;
  }

  gflops = sumgf/niter;
  gflops_exec = sumgf_exec/niter;
  if (niter ==1) 
  {
    sd = 0.0;
    sd_exec = 0.0;
  } else {
    sd = sqrt((sumgf2 - (sumgf*sumgf)/niter)/niter);
    sd_exec = sqrt((sumgf2_exec - (sumgf_exec*sumgf_exec)/niter)/niter);
  }
  
  printf("%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n", 
      sumt/niter, 
      gflops, 
      sd, 
      sumt_exec/niter, 
      gflops_exec, 
      sd_exec, 
      gflops_max, 
      gflops_exec_max );

  free(dA);
}


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{

  kaapic_init(KAAPIC_START_ONLY_MAIN);
  
  // matrix dimension
  int n = 32;
  if (argc > 1)
    n = atoi(argv[1]);

  // block count
  int block_count = 2;
  if (argc > 2)
    block_count = atoi(argv[2]);
    
  // Number of iterations
  int niter = 1;
  if (argc >3)
    niter = atoi(argv[3]);

  // Make verification ?
  int verif = 0;
  if (argc >4)
    verif = atoi(argv[4]);
  
  //int nmax = n+16*incr;
  printf("size   #threads #bs    time      GFlop/s   Deviation\n");
  for (int k=0; k<niter; ++k )
  {
    //doone_exp( 1<<k, block_count, niter, verif );
    doone_exp( n, block_count, niter, verif );
  }

  kaapic_finalize();

  return 0;
}
