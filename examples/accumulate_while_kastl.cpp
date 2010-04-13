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
#include "kastl/numeric"
#include "random.h"

/* basic op evaluated per iteration
*/
struct Sin {
  typedef double result_type;
  void operator()(double& result, double a) 
  {
    result=1; //sin(a);
    usleep(1000);
  }
};

/*
*/
struct Accumulator {
  void operator()(double& result, double value) 
  {
    result += value;
  }
};

/*
*/
struct Predicate {
  Predicate( double v) : value(v) {}
  double value;
  bool operator()(const double& result) 
  {
    return result < value;
  }
};




/* The main thread
*/
int main(int argc, char** argv)
{
  int l, n, iter;
  double t0, t1;
  double avrg = 0;
  double threshhold = 0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) iter = atoi(argv[2]);
  else iter = 1;
  if (argc >3) threshhold = atof(argv[3]);
  else threshhold = 1;

  t0 = kaapi_get_elapsedtime();
  double* input  = new double[10+n];
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
  for(int i = 0; i < 10+n; ++i) 
  {
    input[i] = i; //random.genrand_real3();
  }
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time init:" << t1-t0 << std::endl;


  Sin op;
  Accumulator acc;
  Predicate pred(threshhold);
  double result = 0;
  
  size_t size_eval;
  t0 = kaapi_get_elapsedtime();
  for (l=0; l<iter; ++l)
  {
    result = 0;
    size_eval = kastl::accumulate_while( 
        result, 
        input,
        input+n,
        op,
        acc,
        pred,
        1, 1,
        10
      );
    std::cerr << "Used: " << size_eval << " evaluation(s)" << std::endl;
  }
  t1 = kaapi_get_elapsedtime();
  avrg = (t1-t0)/ (double)iter;

  // Verification of the output
  double refvalue = 0;
  size_t imin_seq = -1UL;
  for (size_t i=0; (i<size_eval) && pred(refvalue); ++i)
  {
    Sin::result_type value;
    op( value, input[i] );
    acc(refvalue, value );
    if (!pred(refvalue) && (imin_seq!=(size_t)-1)) imin_seq = 1+i;
  }
  if (imin_seq ==(size_t)-1) imin_seq = size_eval;
  
  std::cout << "Time       : " << avrg << std::endl;
  std::cout << "Diff       : " << result-refvalue << std::endl;
  std::cout << "Extra evals: " << size_eval - imin_seq << std::endl;

  delete [] input;

  return 0;
}
