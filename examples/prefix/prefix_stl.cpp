/*
 *  test_transform.cpp
 *  ckaapi
 *
 *  Created by TG on 18/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */

#include <sys/time.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <numeric>
#include <iomanip>


/** Return the number of seconds + micro seconds since the epoch
*/
inline double gettime()
{
  struct timeval tp;
  gettimeofday(&tp, 0);
  return (double)(tp.tv_sec) + (double)(tp.tv_usec)*1e-6;
}


struct DoubleAdd {
  double operator()( const double& a, const double& b)
  {
//    volatile double d = sin(a);
//    for (int i=0; i<100; ++i) d = d+sin(a);

    return a+b;
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  int i, cpu, l, n, iter;
  double t0;
  double avrg = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  t0 = gettime();
  double* input __attribute__((aligned (64)))= new double[n];
  double* output __attribute__((aligned (64)))= new double[n];
  t0 = gettime() - t0;

  std::cout << "Time allocate:" << t0 << std::endl;

  t0 = gettime();
  for(i = 0; i < n; ++i) {
    input[i] = i;
    output[i] = 0;
  }
  t0 = gettime() - t0;
  std::cout << "Time init:" << t0 << std::endl;

  DoubleAdd op;

  t0 = gettime();
  for (l=0; l<iter; ++l)
  {
    std::partial_sum(input, input+n, output, op );
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
    p = op(p, double(i));
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
    std::cout << "Result-> cpu:" << cpu << "  :size: " << n << "  :time: " << avrg << std::endl;
  }
  else std::cout << "Verification failed" << std::endl;

  return 0;
}
