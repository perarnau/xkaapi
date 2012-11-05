/*
 *  test_partial_sum.cpp
 *  ckaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
 *
 */
 
//#define LOG


#include "kaapi_adapt.h"
#include "kaapi_impl.h"
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <functional>

  double avrg = 0;
  double start_time;

/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}

#include "prefix.h"


struct DoubleAdd {
  double operator()( const double& a, const double& b)
  {
//    volatile double d = sin(a);
//    for (int i=0; i<10; ++i) d = d+sin(a);
    return a+b;
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  start_time = gettime();
  int cpu, i, l, n, iter;
  double t0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  /* set the concurrency number */
  kaapi_adapt_setconcurrency( cpu );

  t0 = gettime();
  double* input  = new double[n];
  double* output = new double[n];
  t0 = gettime() - t0;
  std::cout << "Time allocate:" << t0 << std::endl;

  /* */
  std::cout << gettime()-start_time << "::Main thread has stack id:" << kaapi_adapt_current_stack()->_index << std::endl;

  t0 = gettime();
  for(i = 0; i < n; ++i) {
    input[i] = i; 
    output[i] = 0;
  }
  t0 = gettime() - t0;
  std::cout << gettime()-start_time << "::Time init:" << t0 << std::endl;

  kaapi_steal_stack_t* current_stack = kaapi_adapt_current_stack();
  
  /* Declare the context to steal work of the sequential computation
     The stack size for storing steal requests (both input and output) is at most 4096 bytes.
     In case of steal request with no enough memory, then the steal request fails.
  */
  kaapi_steal_context_declare( stealcontext, CKAAPI_STEALCONTEXT_STACKSIZE, malloc(CKAAPI_STEALCONTEXT_STACKSIZE) );

  /* push the context to be visible from outside control */
  kaapi_steal_context_push( current_stack, &stealcontext);
  usleep( 1000 );

  DoubleAdd op;

  t0 = gettime();
  for (l=0; l<iter; ++l)
  { 
    partial_sum( &stealcontext, input, input+n, output, op );
  }
  t0 = (gettime() -t0);
  avrg += t0;
  avrg /= (double)iter;
    
  
  if (n <= 20) 
  {
    std::cout << "Values: ";
    for (i=0; i<n; ++i) std::cout << std::setw(4) << input[i];
    std::cout << std::endl;
    std::cout << "Prefix: ";
    for (i=0; i<n; ++i) std::cout << std::setw(4) << output[i];
    std::cout << std::endl;
  }

  // Verification of the output
  bool isok = true;
  
  double p = input[0];
  if (output[0] != p) {
    std::cout << "Fail, prefix computed at index:[" << 0 << "]=" << output[0] << ", should be:" << p << std::endl;
    isok = false;
  }
  for (i=1; (i<n) && isok; ++i)
  {
    p += double(i);
    if (p != output[i] ) 
    {
      std::cout << "Fail, prefix computed at index:[" << i << "]=" << output[i] 
                << ", should be:" << p 
                << ", diff:" << p - output[i] 
                << std::endl;
      /* test all entries isok = false;*/
      isok = false;
    }
    if (input[i] !=i) {
      std::cout << "Waring, input array at index:[" << i << "]=" << input[i] << ", should be:" << i << std::endl;
      isok = false;
    }
  }
  if (isok) {
    std::cout << "Verification ok" << std::endl;
  }
  else std::cout << "Verification failed" << std::endl;
    std::cout << "Result-> cpu:" << cpu << "  :size: " << n << "  :time: " << avrg << std::endl;

  _exit(1);

  /* Pop the steal context
  */
  kaapi_steal_context_pop( current_stack );
  
  /* Destroy the context 
  */
  kaapi_steal_context_destroy( &stealcontext );

  return 0;
}
