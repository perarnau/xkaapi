/*
 *  test_inner_product.cpp
 *  xkaapi
 *
 *  Created by TD on Avril 09.
 *  Updated by FLM on December 09.
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
#include "inner_product.h"
#include "random.h"


/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}

typedef long val_t;

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
  val_t* input  = new val_t[n];
  val_t* input2  = new val_t[n];
  t0 = gettime() - t0;

  val_t res = val_t(0); // result of the computation 

  std::cout << "Time allocate:" << t0 << std::endl;
  Random random(42 + iter);

  usleep(100000);

#if 1
  double* ti = new double[iter];
  t0 = gettime();
  //suite de valeurs necessaires pour le calcul de ln(2)
  for(int i = 0; i < n; ++i) {
    input[i] = val_t(i+1) ;
    input2[i] = val_t(i+1);
  }

  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;


  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    //t0 = gettime();
    res = inner_product(input, input+n, input2, val_t(0));
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
   val_t res_stl = std::inner_product(input, input+n, input2, val_t(0));
   std::cout << "res_stl = " << res_stl << std::endl;

   const val_t diff = res - res_stl;
   bool are_same = false;

   if (diff < 1e-6 && diff > -1e-6)
     are_same = true;

   if (are_same)
     std::cout << "Verification OK!!!!" << std::endl; 
   else
     std::cout << "Verification failed, KO!!!!!!!!!!!!" << std::endl;
   //std::cout.setf(0, std::ios_base::floatfield);

#else
  double avrg_init = 0;
  for (l=0; l<iter; ++l)
  {
    t0 = gettime();
    for(int i = 0; i < n; ++i) {
      input[i] = i;
      input2[i] = i;
    }
    t0 = gettime() - t0;
    avrg_init += t0;

    t0 = gettime();
    res = inner_product(input, input+n, input2, val_t(20));
    //std::cout << "t[" << l << "]=" << t0 << std::endl;

    t0 = (gettime() -t0);
    avrg += t0;

   // Verification of the result
   std::cout << "res = " << res << std::endl;
   if(res==std::inner_product(input, input+n, input2, val_t(20))) std::cout << "Verification OK!!!!" << std::endl;
   else std::cout << "Verification failed, KO!!!!!!!!!!!!" << std::endl;

  }
  avrg /= (double)iter;
  std::cout << "Time init:" << avrg_init/iter << std::endl;
#endif

  std::cout << "Result-> cpu:" << cpu << "  size: " << n << "  time: " << avrg << std::endl;

  delete [] input;
  delete [] input2;

  return 0;
}
