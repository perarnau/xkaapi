/*
 *  test_min_element.cpp
 *  ckaapi
 *
 *  Created by TD on Avril 09.
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
#include "min_element2.h"
#include "timing.h"
#include "initialisation.h"

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

  val_t* input  = initialisation (n, 1);

  val_t res; // result of the computation 
  int steal_count;

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
    
    res = *min_element(&stealcontext, input, input+n, steal_count);
  }

  GET_TICK(tt2);
  std::cout << "PIPOPIPO taille :: " << n << " ncpu :: " << cpu << " iter :: " << iter << " time :: " << TIMING_DELAY (tt1, tt2)/1000000/iter << " ticks :: " << TICK_DIFF (tt1, tt2) << std::endl;

  // Verification of the result
  std::cout << "res = " << res << std::endl;
  if(res==*std::min_element(input, input+n)) std::cout << "Verification OK!!!!" << std::endl; 
  else std::cout << "Verification failed, KO!!!!!!!!!!!!" << std::endl;

  delete [] input;

  //exit(1);
  /* Pop the steal context
  */
  kaapi_steal_context_pop( current_stack );
  
  /* Destroy the context 
  */
  kaapi_steal_context_destroy( &stealcontext );

  return 0;
}
