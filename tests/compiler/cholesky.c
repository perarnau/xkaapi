/*
* Copyright (c) 2008, BSC (Barcelon Supercomputing Center)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the <organization> nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY BSC ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cblas.h>

//----------------------------------------------------------------------------------------------

#pragma css task input(NB) inout(A[NB][NB]) highpriority 
void smpSs_spotrf_tile(float *A,unsigned long NB)
{
unsigned char LO='L';
int INFO;
int nn=NB;
   spotrf_(&LO,
          &nn,
          A,&nn,
          &INFO);
}

#pragma css task input(A[NB][NB], B[NB][NB], NB) inout(C[NB][NB])
void smpSs_sgemm_tile(float  *A, float *B, float *C, unsigned long NB)
{
unsigned char TR='T', NT='N';
float DONE=1.0, DMONE=-1.0;

//    sgemm_2(&NT, &TR,       /* TRANSA, TRANSB */
//           &NB, &NB, &NB,   /* M, N, K        */
//           &DMONE,          /* ALPHA          */
//           A, &NB,          /* A, LDA         */
//           B, &NB,          /* B, LDB         */
//           &DONE,           /* BETA           */
//           C, &NB);         /* C, LDC         */

 // using CBLAS
    cblas_sgemm(
        CblasColMajor,
        CblasNoTrans, CblasTrans,
        NB, NB, NB,
        -1.0, A, NB,
              B, NB,
         1.0, C, NB);


}

#pragma css task input(T[NB][NB], NB) inout(B[NB][NB])
void smpSs_strsm_tile(float *T, float *B, unsigned long NB)
{
unsigned char LO='L', TR='T', NU='N', RI='R';
float DONE=1.0;

//  strsm_2(&RI, &LO, &TR, &NU,  /* SIDE, UPLO, TRANSA, DIAG */
//          &NB, &NB,             /* M, N                     */
//         &DONE,                /* ALPHA                    */
//         T, &NB,               /* A, LDA                   */
//         B, &NB);              /* B, LDB                   */

 // using CBLAS
    cblas_strsm(
        CblasColMajor,
        CblasRight, CblasLower, CblasTrans, CblasNonUnit,
        NB, NB,
        1.0, T, NB,
             B, NB);

}

#pragma css task input(A[NB][NB], NB) inout(C[NB][NB])
void smpSs_ssyrk_tile( float *A, float *C, long NB)
{
unsigned char LO='L', NT='N';
float DONE=1.0, DMONE=-1.0;

//    ssyrk_2(&LO, &NT,          /* UPLO, TRANSA */
//           &NB, &NB,           /* M, N         */
//           &DMONE,             /* ALPHA        */
//           A, &NB,             /* A, LDA       */
//           &DONE,              /* BETA         */
//           C, &NB);            /* C, LDC       */

 // using CBLAS
    cblas_ssyrk(
        CblasColMajor,
        CblasLower,CblasNoTrans,
        NB, NB,
        -1.0, A, NB,
         1.0, C, NB);


}

//----------------------------------------------------------------------------------------------


void compute(struct timeval *start, struct timeval *stop, long NB, long DIM, float **A)
{
  #pragma css start 
  gettimeofday(start,NULL);
  for (long j = 0; j < DIM; j++)
  {
    for (long k= 0; k< j; k++)
    {
      for (long i = j+1; i < DIM; i++) 
      {
        // A[i,j] = A[i,j] - A[i,k] * (A[j,k])^t
        smpSs_sgemm_tile( A[i*DIM+k], A[j*DIM+k], A[i*DIM+j], NB);
      }

    }

    for (long i = 0; i < j; i++)
    {
      // A[j,j] = A[j,j] - A[j,i] * (A[j,i])^t
      smpSs_ssyrk_tile( A[j*DIM+i], A[j*DIM+j], NB);
    }

    // Cholesky Factorization of A[j,j]
    smpSs_spotrf_tile( A[j*DIM+j], NB);
      
    for (long i = j+1; i < DIM; i++)
    {
      // A[i,j] <- A[i,j] = X * (A[j,j])^t
      smpSs_strsm_tile( A[j*DIM+j], A[i*DIM+j], NB);
    }
   
  }	
  #pragma css barrier
  gettimeofday(stop,NULL);
}

//--------------------------------------------------------------------------------


static void  init(int argc, char **argv, long *NB_p, long *N_p, long *DIM_p);

float **A;
float * Alin; // private in init

long NB, N, DIM; // private in main

int
main(int argc, char *argv[])
{
  // local vars
 
unsigned char LO='L';
int  INFO;
 
  struct timeval start;
  struct timeval stop;
  double elapsed;

  // application inicializations
  init(argc, argv, &NB, &N, &DIM);
  // compute with CellSs

  compute(&start, &stop, NB, DIM, (void *)A);

int nn=N;
// compute with library
  
//  spotrf(&LO, &nn, Alin, &nn, &INFO);

  elapsed = 1000000.0 * (stop.tv_sec - start.tv_sec);
  elapsed += 1.0*(stop.tv_usec - start.tv_usec);

// time in usecs
  printf ("%e (s)\n", 1e-6*elapsed);
// perfonrmance in MFLOPS
  printf("%f (GFlops)\n", ((0.33*N*N*N+0.5*N*N+0.17*N)/1000.0/elapsed));

	return 0;
}


static void convert_to_blocks(long NB,long DIM, long N, float *Alin, float **A)
{
  for (long i = 0; i < N; i++)
  {
    for (long j = 0; j < N; j++)
    {
      A[j/NB*DIM+i/NB][(i%NB)*NB+j%NB] = Alin[i*N+j];
    }
  }

}


//void slarnv_(long *idist, long *iseed, long *n, float *x);

void fill_random(float *Alin, int NN)
{
  int i;
  for (i = 0; i < NN; i++)
  {
    Alin[i]=((float)rand())/((float)RAND_MAX);
  }
}


static void init(int argc, char **argv, long *NB_p, long *N_p, long *DIM_p)
{
  long ISEED[4] = {0,0,0,1};
  long IONE=1;
  long DIM;
  long NB;

  
  if (argc==3)
  {
    NB=(long)atoi(argv[1]);
    DIM=(long)atoi(argv[2]);
  }
  else
  {
    printf("usage: %s NB DIM\n",argv[0]);
    exit(0);
  }

  // matrix init
  
  long N = NB*DIM;
  long NN = N * N;

  *NB_p=NB;
  *N_p = N;
  *DIM_p = DIM;
  
  // linear matrix
   Alin = (float *) malloc(NN * sizeof(float));

  // fill the matrix with random values
//  slarnv_(&IONE, ISEED, &NN, Alin);
  fill_random(Alin,NN);

  // make it positive definite
  for(long i=0; i<N; i++)
  {
    Alin[i*N + i] += N;
  }
  
  // blocked matrix
  A = (float **) malloc(DIM*DIM*sizeof(float *));
  for (long i = 0; i < DIM*DIM; i++)
     A[i] = (float *) malloc(NB*NB*sizeof(float));

  convert_to_blocks(NB, DIM, N, Alin, (void *)A);
  
}

