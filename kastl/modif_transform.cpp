/*
 *  test_transform.cpp
 *  xkaapi
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
#include "transform.h"
#include "random.h"


/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}


/* basic op
*/
struct Sin {
  double operator()(double a) 
  {
    return 2*a;
    //return sin(a);
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  int cpu, l, n, iter;
  double t0, t1;
  double avrg = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  t0 = gettime();
  double* input  = new double[n];
  double* output  = new double[n];
  t0 = gettime() - t0;

  std::cout << "@input:" << input << std::endl;
  std::cout << "@output:" << output << std::endl;

  std::cout << "Time allocate:" << t0 << std::endl;
  Random random(42 + iter);

  double d = random.genrand_real3();
  t0 = gettime();
  for (int j=0; j<10000; ++j)
    d = sin(d);
  t0 = gettime() - t0;
  std::cout << "Time 10000 sin:" << t0/10000 << std::endl;

  /* */
  std::cout << "Main thread has stack id:" << kaapi_adapt_current_stack()->_index << std::endl;


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

#if 1
  double* ti = new double[iter];
  t0 = gettime();
  for(int i = 0; i < n; ++i) {
    input[i] = random.genrand_real3();
output[i] = 0;
  }
  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;


  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    //t0 = gettime();
    transform(&stealcontext, input, input+n, output, Sin() );
    //t1 = gettime();
    //ti[l] =  t1 - t0;
//    std::cout << l << " \t" << ti[l] << std::endl;
  //  std::cout << "t[" << l << "]=" << t0 << std::endl;
  }
  t0 = (gettime() -t0);
#if 0
  avrg =0;
  for (l=0; l<iter; ++l)
  {
//    std::cout << l << " \t" << ti[l] << std::endl;
    avrg += ti[l];
  }
#else
  avrg = t0;
#endif
  avrg /= (double)iter;
  delete [] ti;

  // Verification of the output
  bool isok = true;
  Sin op;
  for (int i=0; i<n; ++i)
  {
    if (output[i] !=  op( input[i]))
    {
      std::cout << "Fail, i:" << i << ", @:" << output + i << ", input @:" << input + i << std::endl;
      isok = false;
    }
  }
  if (isok) std::cout << "Verification ok" << std::endl;
#else
  double avrg_init = 0;
  for (l=0; l<iter; ++l)
  {
    t0 = gettime();
    for(int i = 0; i < n; ++i) {
      input[i] = random.genrand_real3();
    }
    t0 = gettime() - t0;
    avrg_init += t0;

    t0 = gettime();
    transform(input, input+n, output, Sin() );
  //  std::cout << "t[" << l << "]=" << t0 << std::endl;

    t0 = (gettime() -t0);
    avrg += t0;

    // Verification of the output
    bool isok = true;
    Sin op;
    for (int i=0; i<n; ++i)
    {
      if (output[i] !=  op( input[i]))
      {
        std::cout << "Fail, i:" << i << ", @:" << output + i << ", input @:" << input + i << std::endl;
        isok = false;
        exit(1);
      }
    }
  }
  avrg /= (double)iter;
  std::cout << "Time init:" << avrg_init/iter << std::endl;
#endif

  std::cout << "Result-> cpu:" << cpu << "  size: " << n << "  time: " << avrg << std::endl;

  delete [] input;
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
