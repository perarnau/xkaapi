/*
 *  test_transform.cpp
 *  xckaapi
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
#include <algorithm>
#include "random.h"

/**
*/
double kaapi_get_elapsedtime(void)
{
  struct timeval tv;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
}


/* basic op
*/
struct Sin {
  double operator()(double a) 
  {
    return sin(a);
  }
};


struct Op2 {
    Op2(){ };
    double operator()(double a){
         return  2*a;
    }
};



/* The main thread
*/
int main(int argc, char** argv)
{
  int l, n, iter;
  double t0, t1;
  double avrg = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) iter = atoi(argv[2]);
  else iter = 1;

  t0 = kaapi_get_elapsedtime();
  double* input  = new double[n];
  t1 = kaapi_get_elapsedtime();

  std::cout << "Time allocate:" << t1-t0 << std::endl;
  Random random(42 + iter);

  double d = random.genrand_real3();
  t0 = kaapi_get_elapsedtime();
  for (int j=0; j<10000; ++j)
    d = sin(d);
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time 10000 sin:" << (t1-t0)/10000 << std::endl;

  t0 = kaapi_get_elapsedtime();
  for(int i = 0; i < n; ++i) 
  {
    input[i] = i; //random.genrand_real3();
  }
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time init:" << t1-t0 << std::endl;


//  Sin op;
  Op2 op;
  
  volatile int cnt = 0;
  t0 = kaapi_get_elapsedtime();
  for (l=0; l<iter; ++l)
  {
    std::for_each(input, input+n, op );
#if 0
    std::for_each( input, input+n, [&](double val) {
      op(val);
      ++cnt;
    } );
#endif
  }
  t1 = kaapi_get_elapsedtime();
  avrg = (t1-t0)/ (double)iter;

  std::cout << "Result-> size: " << n << "  time: " << avrg << ", cnt:" << cnt << std::endl;

  delete [] input;

  return 0;
}
