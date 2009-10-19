/*
 *  test_merge.cpp
 *  xkaapi
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
#include "merge.h"
#include "random.h"


/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}

typedef double val_t;

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
  double t0, t1;
  double avrg = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  t0 = gettime();
  val_t* input  = new val_t[n];
  val_t* input2  = new val_t[n];
  val_t* output  = new val_t[2*n];
  t0 = gettime() - t0;

  std::cout << "Time allocate:" << t0 << std::endl;
  Random random(42 + iter);


  /* */
  std::cout << "Main thread has stack id:" << kaapi_adapt_current_stack()->_index << std::endl;


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

#if 1
  double* ti = new double[iter];
  t0 = gettime();
  for(int i = 0; i < n; ++i) {
    input[i] = i+1; /*random.genrand_real3();*/
    input2[i] = 2*(i+1); /*random.genrand_real3();*/
  }

  //std::sort(input, input+n);
  //std::sort(input2, input2+n);

  int size_output = 2*n;
  for(int i = 0; i < size_output; i++) output[i] = val_t(0);

  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;


  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    //t0 = gettime();
    merge(&stealcontext, input, input+n, input2, input2+n, output);
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

   // Verification of the result
   is_sorted(output, (2*n));

#else
  double avrg_init = 0;
  for (l=0; l<iter; ++l)
  {
    t0 = gettime();
    for(int i = 0; i < n; ++i) {
      input[i] = random.genrand_real3();
      input2[i] = random.genrand_real3();
    }
   std::sort(input, input+n);
   std::sort(input2, input2+n);

    int size_output = 2*n;
    for(int i = 0; i < size_output; i++) output[i] = 0;

    t0 = gettime() - t0;
    avrg_init += t0;

    t0 = gettime();
    merge(input, input+n, input2, input2+n, output);
    //std::cout << "t[" << l << "]=" << t0 << std::endl;

    t0 = (gettime() -t0);
    avrg += t0;

   // Verification of the result
   is_sorted(output, (2*n));

  }
  avrg /= (double)iter;
  std::cout << "Time init:" << avrg_init/iter << std::endl;
#endif

  std::cout << "Result-> cpu:" << cpu << "  size: " << n << "  time: " << avrg << std::endl;

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