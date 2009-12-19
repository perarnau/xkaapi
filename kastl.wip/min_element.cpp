/*
 *  test_min_element.cpp
 *  xkaapi
 *
 *  Created by TD on Avril 09.
 *  Updated by FLM on Decembre 09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include "kaapi.h"
#include "min_element.h"
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


/* The main thread
*/
int main(int argc, char** argv)
{
  int cpu, l, n, iter;
  double t0;
  double avrg = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  t0 = gettime();
  double* input  = new double[n];
  t0 = gettime() - t0;

  double res = 0; // result of the computation 

  std::cout << "Time allocate:" << t0 << std::endl;
  Random random(42 + iter);

  usleep(100000);

#if 1
  double* ti = new double[iter];
  t0 = gettime();
  for(int i = 0; i < n; ++i) {
    input[i] = random.genrand_real3();
  }

  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;

  std::cout << "Initial min_element = " << *input << std::endl;

  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    
    //t0 = gettime();
    res = *min_element(input, input+n);
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
   std::cout << "res = " << res << std::endl;
   if(res==*std::min_element(input, input+n)) std::cout << "Verification OK!!!!" << std::endl; 
   else std::cout << "Verification failed, KO!!!!!!!!!!!!" << std::endl;


#else
  double avrg_init = 0;
  for (l=0; l<iter; ++l)
  {
    t0 = gettime();
    for(int i = 0; i < n; ++i) {
      input[i] = val_t(20);
    }
    t0 = gettime() - t0;
    avrg_init += t0;

    t0 = gettime();
    res = min_element(input, input+n);
    //std::cout << "t[" << l << "]=" << t0 << std::endl;

    t0 = (gettime() -t0);
    avrg += t0;

   // Verification of the result
   std::cout << "res = " << res << std::endl;
   if(res==*std::min_element(input, input+n)) std::cout << "Verification OK!!!!" << std::endl;
   else std::cout << "Verification failed, KO!!!!!!!!!!!!" << std::endl;

  }
  avrg /= (double)iter;
  std::cout << "Time init:" << avrg_init/iter << std::endl;
#endif

  std::cout << "Result-> cpu:" << cpu << "  size: " << n << "  time: " << avrg << std::endl;

  delete [] input;

  return 0;
}
