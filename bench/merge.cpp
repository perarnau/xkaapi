/*
 *  test_merge.cpp
 *  ckaapi
 *
 *  Created by TD on Avril 09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#define BENCH_MERGE
#include "kaapi_adapt.h"
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include "merge.h"
#include "timing.h"
#include "initialisation.h"

template<class T>
void is_sorted(T* output, int sz)
{
     bool is_sorted = true;
     for(int i=1; i < sz; i++) {
         if(output[i-1] > output[i]) {
             std::cout << "output[" << i-1 << "]=" << output[i-1] << " is not less than  ";
             std::cout << "output[" << i << "]=" << output[i] << ", " << std::endl;
             std::cout << "Then the table is not sorted well" << std::endl;
             is_sorted = false;
           break;
        }
     }
    if(is_sorted) std::cout << "0:The table has been well sorted..........";
    std::cout << std::endl;
}


/* The main thread
*/
int main(int argc, char** argv)
{
  int cpu, l, n, iter;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  val_t* input   = initialisation (n, 1);
  val_t* input2  = initialisation (n, 2);
  val_t* output  = initialisation (2*n, 0);

  kaapi_steal_stack_t* current_stack = kaapi_adapt_current_stack();
  
  /* Declare the context to steal work of the sequential computation
     The stack size for storing steal requests (both input and input) is at most 4096 bytes.
     In case of steal request with no enough memory, then the steal request fails.
  */
  kaapi_steal_context_declare( stealcontext, CKAAPI_STEALCONTEXT_STACKSIZE, malloc(CKAAPI_STEALCONTEXT_STACKSIZE) );

  /* push the context to be visible from outside control */
  kaapi_steal_context_push( current_stack, &stealcontext);

  /* set the concurrency number */
  kaapi_adapt_setconcurrency( cpu );
  
  usleep(100000);

  timing_init ();
  
  tick_t tt1, tt2;
  GET_TICK (tt1);
  for (l=0; l<iter; ++l)
  {
    merge(&stealcontext, input, input+n, input2, input2+n, output);
  }
  GET_TICK(tt2);
  std::cout << "PIPOPIPO taille :: " << n << " ncpu :: " << cpu << " iter :: " << iter << " time :: " << TIMING_DELAY (tt1, tt2)/1000000/iter << " ticks :: " << TICK_DIFF (tt1, tt2) << std::endl;
  
  // Verification of the result
  is_sorted(output, (2*n));

  delete [] input;
  delete [] input2;
  delete [] output;

  //exit(1);
  /* Pop the steal context
  */
  kaapi_steal_context_pop( current_stack );
  
  /* Destroy the context 
  */
  kaapi_steal_context_destroy( &stealcontext );

  return 0;
}
