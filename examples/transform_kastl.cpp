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
#include "kastl/algorithm"
#include "random.h"

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
    double operator()(double a)
    {
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
  double* output  = new double[n];
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
    output[i] = 0;
  }
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time init:" << t1-t0 << std::endl;


//  Sin op;
  Op2 op;
  
  t0 = kaapi_get_elapsedtime();
  for (l=0; l<iter; ++l)
  {
    kastl::transform(input, input+n, output, op );
  }
  t1 = kaapi_get_elapsedtime();
  avrg = (t1-t0)/ (double)iter;

  // Verification of the output
  bool isok = true;
  for (int i=0; i<n; ++i)
  {
    if (output[i] !=  op( input[i]))
    {
      std::cout << "Fail, i:" << i << ", @:" << output + i << ", input @:" << input + i 
                << " -> result:" << output[i] << " correct is:" << op(input[i]) << std::endl;
      isok = false;
    }
  }
  if (isok) std::cout << "Verification ok" << std::endl;

  std::cout << "Result-> size: " << n << "  time: " << avrg << std::endl;

  delete [] input;
  delete [] output;

  return 0;
}
