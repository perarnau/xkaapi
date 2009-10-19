/*
 *  clement_problem.cpp
 *  xkaapi
 *  Exemple on how to iterate through a vector with a sliding window of constant size.
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#define BLOC_SIZE_I 32

/** Compute A^k x where A is a dense matrix, x a bloc of columns.
    
    Structure of the dense matrix is assumed to be a rowmajor representation (C like).
    The notations are :
      - lda  : leading dimension of A (number of colums of A)
      - i, j : index of A[i,j]
      - N, M : the shape of A (number of rows, number of columns)
    The code make an explicit call to C Blas Dgemm function : 
        void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const double alpha, const double *A,
                     const int lda, const double *B, const int ldb,
                     const double beta, double *C, const int ldc);

    The parallelization scheme is based on a decomposition of the matrix by rows with
    a synchronization at the end of each matrix-vector step.
    
    I ASSUME THAT ALL DIVISION ARE INTEGER DIVISION
    
    I begin to write this version because it is less cache consumming than spliting
    x is sub-columns...
    
    Next version will provided a two level parallelisation : the first level describes
    in this code and a decomposition of each bloc of rows of A in bloc of columns.
*/
class MatrixVectorProductWork : public kaapi_work_t {
public:
  /* cstor */
  MatrixVectorProductWork(
    int K, 
    double* A, int lda, int N, int M,         /* A of size N x M */
    double* X, int ldX, int L,                /* X of size M x L */
    double* Y, int ldY                        /* Y of size N x L */
  ) : _A(A), _lda(lda), _N(N), _M(M),
      _X(X), _ldX(ldX), _L(L)
      _Y(Y), _ldY(ldY)
  {
  }
  
  /* do computation */
  void doit();

protected:  
  
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, kaapi_work_t* data)
  {
    MatrixVectorProductWork* w = (MatrixVectorProductWork*)data;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_task_t* self_task, int count, kaapi_steal_request_t** request, 
                        MatrixVectorProductWork* w, int ibegin, int* N
                      )
  {
    int i = 0;

    size_t blocsize;
    int thief_end = *N;

    MatrixVectorProductWork* output_work =0; 

    /* threshold should be defined (...) */
    if ((*N - ibegin) < 1) goto reply_failed;

    /* keep a bloc of rows for myself */
    blocsize = (*N-ibegin) / (1+count); 
    
    /* adjust the number of bloc in order to do not have bloc of size less than 1 */
    if (blocsize < 1) { count = (*N-ibegin)-1; blocsize = 1; }
    
    /* reply to all thiefs */
    while (count >0)
    {
      /* generate work for blocsize rows between [thief_end..thief_end - blocsize-1] */
      if (request[i] !=0)
      {
        output_work->_A   = w->_A+(thief_end-blocsize)*w->_ldA;
        output_work->_lda = w->_ldA;
        output_work->_N   = blocsize;
        output_work->_M   = w->_M;
        output_work->_X   = w->_X;
        output_work->_ldX = w->_ldX;
        output_work->_Y   = w->_Y+(thief_end-blocsize)*w->_ldY;
        output_work->_ldY = w->_ldY;
        
        thief_end -= blocsize;

        /* reply ok (1) to the request */
        kaapi_request_reply_ok( self_task, request[i], 
                &thief_entrypoint, 
                output_work, sizeof(output_work), 
                KAAPI_MASTER_FINALIZE_FLAG
        );
        --count; 
      }
      ++i;
    }
  /* mute the end of input work of the victim */
  *N  = thief_end;
  return;
      
reply_failed: /* to all other request */
    while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed to the request */
        kaapi_request_reply_fail( request[i] );
        --count; 
      }
      ++i;
    }
  }


/** Main entry point
*/
void MatrixVectorProductWork::doit()
{
  kaapi_task_t* self_task = kaapi_self_task();

  int N = _N;
  int K = _K;

  for (int k=0; k<K; ++k)
  {
    
    /* here all the following code is a bloc-matrix-vector product that could be replaced by
       a call to cblas_dgemm on the whole matrix A, vector X
    */
    for (int i=0; i<N; i += BLOC_SIZE_I)
    {
      /* definition of the steal point where steal_work may be called in case of steal request 
         note that here local_iend is passed as parameter and updated in case of steal.
         The splitter cannot give more than WINDOW_SIZE size work to all other threads.
         Thus each thief thread cannot have more than WINDOW_SIZE size work.
      */
      kaapi_stealpoint( self_task, splitter, this, &i, &N );

      cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasRowMajor,
                   _M, BLOC_SIZE_I, _M, 
                   1.0 /* alpha */, 
                   _A+i*_ldA, _ldA,
                   _X+i*_ldX, _ldX,
                   1.0 /* beta */,
                   _Y+i*_ldY
      );
    }
    
    kaapi_finalize_steal();
    
    /* swap X and Y */
    std::swap( _X, _Y );
    std::swap( _ldX, _ldY );
  }
}



/**
*/
void clement_problem ( int K, double* A, int N, int M, double* X, int L, double* Y )
{
  /* push new adaptative task with following flags:
    - master finalization: the main thread will waits the end of the computation.
  */
  kaapi_task_t task;
  kaapi_self_push_task( &task, KAAPI_MASTER_FINALIZE_FLAG );

  MatrixVectorProductWork work( K, A, N, M, X, L, Y);
  work.doit();

  kaapi_self_pop_task();
}
