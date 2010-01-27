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
#include "count_concurrent.h"
#include "random.h"


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

  size_t count0 =0;
  t0 = kaapi_get_elapsedtime();
  for(int i = 0; i < n; ++i) 
  {
    input[i] = i % 127;
    if (input[i] == 0) ++count0;
  }
  t1 = kaapi_get_elapsedtime();
  std::cout << "Time init:" << t1-t0 << std::endl;


  t0 = kaapi_get_elapsedtime();
  for (l=0; l<iter; ++l)
  {
    size_t retval = count(input, input+n, 0 );
    kaapi_assert_debug(retval == count0);
  }
  t1 = kaapi_get_elapsedtime();
  avrg = (t1-t0)/ (double)iter;

  std::cout << "Result-> size: " << n << "  time: " << avrg << std::endl;

  delete [] input;

  return 0;
}
