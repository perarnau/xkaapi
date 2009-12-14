/*
 *  test_generate.cpp
 *  xkaapi
 *
 *  Created by TD on Avril 09.
 *  Udpated by FLM on December 09.
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
#include "generate.h"
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

///////////////////////////////////////////////////////////////////
/*
Copied from : http://www.cplusplus.com/reference/algorithm/generate.html
*/

// class generator:
struct c_unique {
  size_t current;
  c_unique() {current=0;}
  size_t operator()() {return /*++current*/current+10;}
} UniqueNumber;
/////////////////////////////////////////////////////////////////////


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
  double* output  = new double[n];
  t0 = gettime() - t0;

  std::cout << "Time allocate:" << t0 << std::endl;
  Random random(42 + iter);

  usleep(100000);

#if 1
  double* ti = new double[iter];
  t0 = gettime();
  for(int i = 0; i < n; ++i) {
    output[i] = 0;
  }
  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;


  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    //t0 = gettime();
    generate(output, output+n, UniqueNumber);
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
  for (int i=0; i<n; ++i)
  {
    //std::cout << "output[" << i << "]=" << output[i] << std::endl;
    if (output[i] !=  val_t(10))
    {
      std::cout << "Fail, i:" << i << ", @:" << output + i  << std::endl;
      isok = false;
      //exit(1);
    }
  }
  if (isok) std::cout << "Verification ok" << std::endl;
#else
  double avrg_init = 0;
  for (l=0; l<iter; ++l)
  {
    t0 = gettime();
    for(int i = 0; i < n; ++i) {
      output[i] = 0;
    }
    t0 = gettime() - t0;
    avrg_init += t0;

    t0 = gettime();
    generate(output, output+n, UniqueNumber);
  //  std::cout << "t[" << l << "]=" << t0 << std::endl;

    t0 = (gettime() -t0);
    avrg += t0;

    // Verification of the output
    bool isok = true;
    for (int i=0; i<n; ++i)
    {
      if (output[i] !=  val_t(10))
      {
        std::cout << "Fail, i:" << i << ", @:" << output + i  << std::endl;
        isok = false;
        exit(1);
      }
    }
  }
  avrg /= (double)iter;
  std::cout << "Time init:" << avrg_init/iter << std::endl;
#endif

  std::cout << "Result-> cpu:" << cpu << "  size: " << n << "  time: " << avrg << std::endl;

  delete [] output;

  return 0;
}
