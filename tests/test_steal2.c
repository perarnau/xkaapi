/*
 *  test_steal.c
 *  ckaapi
 *
 *  Created by TG on 04/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include "kaapi_adapt.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


/** The sequential algorithm to compute iterated product
*/
double iterated_product( kaapi_steal_context_t* stack, int size, double* d);


/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}


/** Stucture of a work
*/
typedef struct work {
  int     size;
  double* d;
  double  r;
} work;


/* Entry in case of thief execution */
void thief_iterated_product(kaapi_steal_context_t* sc, void* data)
{
  work* w = (work*)data;
  int size = w->size;
  w->size = 0;
  w->r = iterated_product(sc, size, w->d );
}

/** splitter_work is called within the context of the steal point
*/
void splitter_work( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, int* size, int i, double* d)
{
#if 1||defined(LOG)
  printf("Split work context:: sc:0x%lx, #reqs:%i, sc size:%i\n", (unsigned long)stealcontext, count, *size -i);
#endif
  work* w =0;
  int med, newsize;

  if (*size- i < 512) return;
  
  /* split size .. i in two parts, count-1 requests will have failed status */
  med = (*size+i) / 2;
  newsize = *size-med;

  for (i=0; i<KAAPI_MAXSTACK_STEAL; ++i)
  {
    if (request[i] !=0) 
    {
      /* allocate the work for the thief */
      if (w == 0) 
      {
        if (kaapi_steal_context_alloc_result( stealcontext, request[i], (void**)&w, sizeof(work) ) ==0)
        {
          w->size = newsize;
          w->d = d+med;
          w->r = 0;
          *size = med;
          kaapi_request_reply( request[i], stealcontext, &thief_iterated_product,  1);
#if defined(LOG)
          printf("Split work context:: sc:0x%lx sc size:%i, split size:%i\n\f", (unsigned long)stealcontext, med, newsize );
#endif
#if 1||defined(LOG)
          printf("Split SUCCESS work context:: sc:0x%lx, #reqs:%i, sc size:%i\n", (unsigned long)stealcontext, count, *size -i);
#endif
        }
        else {
          kaapi_request_reply( request[i], stealcontext, 0, 0);
#if 1||defined(LOG)
          printf("Split FAILED, allocation fails,  work context:: sc:0x%lx, #reqs:%i, sc size:%i\n", (unsigned long)stealcontext, count, *size -i);
#endif
        }
      }
      else {
        kaapi_request_reply( request[i], stealcontext, 0, 0);
#if 1||defined(LOG)
        printf("Split FAILED, already allocated result to other request,  work context:: sc:0x%lx, #reqs:%i, sc size:%i\n", (unsigned long)stealcontext, count, *size -i);
#endif
      }
      printf("Split REMAIN requests: #req:%i\n", KAAPI_ATOMIC_READ( &(stealcontext->_list_request.count) ) );
    }
  }
}


/* Called by the victim thread to collect works from all others threads
*/
int reducer( kaapi_steal_result_t* thief, void* data, kaapi_steal_context_t* sc, double* d, double* r )
{
  work* thief_work = (work*)data;
  *r += thief_work->r;
#if defined(LOG)
  printf("Reducer:: Victim work sc:0x%lx, r:%f\n", (unsigned long)sc, *r);
  printf("Reducer:: Thief work:: @=0x%lx real size:%i, size:%i, r:%f\n", (unsigned long)thief_work, thief_work->d-d, thief_work->size, thief_work->r);
#endif
  if (thief_work->size !=0) {
#if defined(LOG)
    printf("Remainder size:%i\n", thief_work->size);
#endif
    *r += iterated_product( sc, thief_work->size, thief_work->d);
  }
  return 0;
}


/* Called on thief thread to give back remaining work after a steal request
*/
int pass_remainder_work(void* thief_input, int i, int size, double* d, double r)
{
  work* input_work = (work*)thief_input;
  input_work->size = size-i;
  input_work->d = d+i;
  input_work->r = r;
  return 1;
}


/** A pure thief thread
*/
void* startup_routine(void* argv)
{
  unsigned int seed = 0;
  unsigned int victim;
  kaapi_steal_context_t* kpsc;
  int size_stack =0;
  void* stackaddr =0;

  printf("Thief started!\n");
  
  /* declare and initialize a request object in order to steal other threads
  */
  kaapi_steal_stack_t* current_stack = kaapi_adapt_current_stack();
  kaapi_thief_request_declare( request );
  kaapi_thief_request_init( &request, size_stack, stackaddr );
  

redo_post:
  /** Select a victim 
  */
  do {
    if (KAAPI_ATOMIC_READ( &kaapi_index_stacksteal ) >0)
      victim = rand_r( &seed ) % KAAPI_ATOMIC_READ( &kaapi_index_stacksteal );
    else 
      victim = 0;
  } while ((kaapi_all_stealprocessor[victim] ==0) || (kaapi_all_stealprocessor[victim] == current_stack) || KAAPI_QUEUE_EMPTY( kaapi_all_stealprocessor[victim] ));


  /** Post a request to steal work
  */
  /* Post non blocking request 
  */
  kpsc = KAAPI_QUEUE_BACK(kaapi_all_stealprocessor[victim]);
  if (kpsc ==0) goto redo_post;
  
  size_stack = 256;
  stackaddr = malloc(256);
  kaapi_thief_post_request( &request, kpsc, size_stack, stackaddr );
  
  /* may do something here */
  
  /* Wait reply 
  */
  kaapi_thief_wait_request( &request );
  
  /* if no sucess retry 
  */
  if (!kaapi_thief_request_ok(&request)) goto redo_post;
    
  /* Do the local computation
  */
  kaapi_thief_execute( &request );
  
#if 0
  w = (work*)kaapi_thief_getcontext( &request );
#if defined(LOG)
  printf("Thief begin with work:: @=0x%lx, size:%i, r=%f\n", (unsigned long)w, w->size, w->r);
#endif
  size = w->size;
  w->size = 0;
  w->r = iterated_product( &request, size, w->d);
#if defined(LOG)
  printf("Thief end with work:: @=0x%lx, size:%i, r=%f\n", (unsigned long)w, w->size, w->r);
#endif
#endif

//  printf("THIEF work, size=%i, r=%f\n", w->size, w->r);

  kaapi_thief_reply( &request );
  
  goto redo_post;
}


/** Adaptive iterated product
*/
double iterated_product( kaapi_steal_context_t* steal_context, int size, double* d)
{
  double* d0;
  int i,k;
  
#if defined(LOG)
  printf("BEGIN: Iterated product: sc:0x%lx d:0x%lx, d[0]:%f, size:%i\n", (unsigned long)steal_context, (unsigned long)d, d[0], size);
#endif
  
  /* apply nop transform over the array */
  work w;
  w.size = size;
  w.d = d;
  w.r = 0;

  for (i=0; i<w.size; )
  {
    /* definition of the finalize point where the current thread should stop because a victim thread requests 
       the remainder work 
    */
//    if (kaapi_finalizepoint( steal_context, pass_remainder_work, i, size, d, r )) return r;

    /* definition of the steal point where steal_work may be called in case of steal request */
    kaapi_stealpoint( steal_context, splitter_work, &w);
    
    /* increase grain for the nano loop */
    d0 = w.d+i;
    if (w.size-i > 512)
    {
      for (k=0; k<512; ++k)
        r += d0[k];
      i += 512;
    }
    else {
      for (k=0; k<size-i; ++k)
        r += d0[k];
      i = size;
    }
  }
#if defined(LOG)
  printf("END: Iterated product: sc:0x%lx d:0x%lx, d[0]:%f, size:%i, result: %f\n", (unsigned long)steal_context, (unsigned long)d, d[0], size, r);
#endif

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( steal_context, reducer, steal_context, d, &r );

  /* Here the thiefs have finish the computation and returns their values which have been reduced
     using reducer function.
   */  
  return r;
}


/* The main thread
*/
int main(int argc, char** argv)
{
  int l,i, size, iter;
  double r; /* result */
  double *d0 =0;
  double t0;
  kaapi_t thread1;
  kaapi_t thread2;
  void* result;


  if (argc >1) size = atoi(argv[1]);
  else size = 10000;
  if (argc >2) iter = atoi(argv[2]);
  else iter = 1;

  /* */
  kaapi_steal_stack_t* main_stack = kaapi_adapt_current_stack();

  /* Declare the context to steal work of the sequential computation
     The stack size for storing steal requests (both input and output) is at most 256 bytes.
     In case of steal request with no enough memory, then the steal request fails.
  */
  kaapi_steal_context_declare( steal_context, 256, alloca(256) );

  /* push the context to be visible from outside control */
  kaapi_steal_context_push( main_stack, &steal_context);


  kaapi_create( &thread1, 0, startup_routine, (void *)123 );

  /** Create a thief 
  */
  kaapi_create( &thread2, 0, startup_routine, (void *)123 );

  d0 = (double*)malloc( size*sizeof(double) );
  for (i=0; i<size; ++i) d0[i] = (double)i;

  t0 = gettime();
  for (l=0; l<iter; ++l)
    r += iterated_product( &steal_context, size, d0 );
  t0 = (gettime() -t0)/(double)iter;

  printf("Result::: size=%i, sum is: %f, time:%f\n", size, r, t0 );

  /* Pop the steal context
  */
  kaapi_steal_context_pop( main_stack );
  
  /* Destroy the context 
  */
  kaapi_steal_context_destroy( &steal_context );

  exit(1);
  kaapi_join( thread1, &result );
  kaapi_join( thread2, &result );
    
  return 0;
}
