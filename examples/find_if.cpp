/*
 *  test_find_if.cpp
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
#include "find_if.h"
#include <assert.h>

struct IsEqual {
  int value;
  IsEqual( int v) : value(v) {}
  bool operator()( const int& d)
  { 
    volatile double r;
    for (int i=0; i<10000; ++i)
      r += sin(d);
    return d == value; 
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  int cpu, i, l, n, value, iter;
  double t0, t1;
  double avrg = 0;
  int* ifound =0;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;
  if (argc >4) value = atoi(argv[4]);
  else value = 1;

  t0 = kaapi_get_elapsedtime();
  int* input = new int[1+n];
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time allocate:" << t1-t0 << std::endl;

  t0 = kaapi_get_elapsedtime();
  for(i = 0; i <= n; ++i) {
    input[i] = i;
  }
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time init:" << t1-t0 << std::endl;

  IsEqual op(value);

  t0 = kaapi_get_elapsedtime();
  for (l=0; l<iter; ++l)
  {
    ifound = find_if( input, input+n, op );
    std::cout << "ifound [" << l << "]=" << ifound << ", value=" << *ifound << std::endl;
  }
  t1 = kaapi_get_elapsedtime();
  avrg = (t1-t0)/(double)iter;
  std::cout << "Adaptive find_if find: iterator:" << ifound << std::endl;

  // Verification of the output
  bool isok = true;
  int* istl_found = std::find_if( input, input + n, op );
  if  (ifound != istl_found)
  {
    std::cout << "Fail, found predicate at i:" << i << ", @:" << istl_found << ", value=" << *istl_found << ", my find_if @:" << ifound << std::endl;
    isok = false;
  }
  if (isok) std::cout << "Verification ok, iterator:" << ifound << std::endl;
  else std::cout << "Verification failed" << std::endl;

  std::cout << "Result-> cpu:" << cpu << "  :size: " << n << "  :time: " << avrg << std::endl;

  return 0;
}
