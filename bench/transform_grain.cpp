/*
 *  test_transform.cpp
 *  ckaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#include "kaapi_adapt.h"
#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include "transform_grain.h"
#include "timing.h"
#include "initialisation.h"

/* basic op
*/
struct Sin {
  val_t operator()(val_t a) 
  {
    return 2*a;
    //return sin(a);
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  timing_init ();

  int cpu, l, n, iter, grain;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;
  if (argc >4) grain = atoi(argv[4]);
  else grain = 512;

  Sin op;

  val_t* input  = initialisation (n, 1);
  val_t* output  = initialisation (n, 0);

  kaapi_steal_stack_t* current_stack = kaapi_adapt_current_stack();
  
  /* Declare the context to steal work of the sequential computation
     The stack size for storing steal requests (both input and output) is at most 4096 bytes.
     In case of steal request with no enough memory, then the steal request fails.
  */
  kaapi_steal_context_declare( stealcontext, CKAAPI_STEALCONTEXT_STACKSIZE, malloc(CKAAPI_STEALCONTEXT_STACKSIZE) );

  /* push the context to be visible from outside control */
  kaapi_steal_context_push( current_stack, &stealcontext);

  /* set the concurrency number */
  kaapi_adapt_setconcurrency( cpu );
  
  usleep(100000);

  tick_t tt1, tt2;
  GET_TICK (tt1);

  for (l=0; l<iter; ++l)
  {
    transform(&stealcontext, input, input+n, output, op, grain);
  }

  GET_TICK(tt2);

  std::cout << "PIPOPIPO taille :: " << n << " ncpu :: " << cpu << " iter :: " << iter << " time :: " << TIMING_DELAY (tt1, tt2)/1000000/iter << " ticks :: " << TICK_DIFF (tt1, tt2) << std::endl;



  // Verification of the output
  bool isok = true;
  for (int i=0; i<n; ++i)
  {
    if (output[i] !=  op( input[i]))
    {
      std::cout << "Fail, i:" << i << ", @:" << output + i << ", input @:" << input + i << std::endl;
      isok = false;
    }
  }
  if (isok) std::cout << "Verification ok" << std::endl;

  delete [] input;
  delete [] output;

  /* Pop the steal context
  */
  kaapi_steal_context_pop( current_stack );
  
  /* Destroy the context 
  */
  kaapi_steal_context_destroy( &stealcontext );

  return 0;
}
