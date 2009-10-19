/*
 *  test_steal.c
 *  ckaapi
 *
 *  Created by TG on 04/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi_adapt.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}


/** splitter_work is called within the context of the steal point
   Return value 0 in case of failure of the steal operation else !=0 in case of success
*/
int splitter_work( kaapi_steal_context_t* stealrequest, int count, kaapi_steal_request_t** request, int* size, int i, double* d)
{
  return 1;
}

/*
*/
int reducer( kaapi_steal_result_t* thief, void* data, double* r )
{
  return 0;
}


/*
*/
int main(int argc, char** argv)
{
  int i, l, k, size, iter;
  double r; /* result */
  double *d, *d0 =0;
  double t0;

  if (argc >1) size = atoi(argv[1]);
  else size = 10000;
  if (argc >2) iter = atoi(argv[2]);
  else iter = 10;

  /* */
  kaapi_steal_stack_t* main_stack = kaapi_adapt_current_stack();

  /* declare the context to steal iterator product */
  kaapi_steal_context_declare( steal_context, 0, 0 );

  /* push the context to be visible from outside control */
  kaapi_steal_context_push( main_stack, &steal_context)
  
  d0 = (double*)malloc( size*sizeof(double) );
  for (i=0; i<size; ++i) d0[i] = (double)i;

  t0 = gettime();
  for (l = 0; l<iter; ++l)
  {
    d = d0;
    
    /* apply nop transform over the array */
    for (i=0; i<size; i+= 512, d += 512)
    {
      /* definition of the steal point where steal_work may be called in case of steal request */
      kaapi_stealpoint( &steal_context, splitter_work, &size, i, d );
      if ( size - i < 512) {
        for (k=0; k<size-i; ++k)
          r += sin(d[k]);
      }
      else {
        for (k=0; k<512; ++k)
          r += sin(d[k]);
      }
    }
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( &steal_context, reducer, &r );
  /* Here the thiefs have finish the computation.
     Moreover, the sequential computation is finish, because no finalisation code is required.
   */
  
  t0 = (gettime() -t0)/(double)iter;

  kaapi_steal_context_pop( main_stack );
  
  kaapi_steal_context_destroy( &steal_context );

  printf("Result::: size=%i, sum is: %f, time:%f\n", size, r, t0 );

  return 0;
}
