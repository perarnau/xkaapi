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
#include <algorithm>
#include "transform.h"
#include "timing.h"
#include "initialisation.h"

/* basic op
*/
struct Op {
  val_t operator()(val_t a) 
  {
    return 2*a;
    //return sin(a);
  }
};


/* The main thread
*/
int main(int argc, char** argv)
{
  timing_init ();

  int cpu, l, n, iter;

  if (argc >1) n = atoi(argv[1]);
  else n = 10000;
  if (argc >2) cpu = atoi(argv[2]);
  else cpu = 1;
  if (argc >3) iter = atoi(argv[3]);
  else iter = 1;

  Op op;

  val_t* input  = initialisation (n, 1);
  val_t* output  = initialisation (n, 0);

  usleep(100000);

  tick_t tt1, tt2;
  GET_TICK (tt1);

  for (l=0; l<iter; ++l)
  {
    transform( input, input+n, output, op );
  }

  GET_TICK(tt2);

  std::cout << "PIPOPIPO taille :: " << n << " ncpu :: " << cpu << " iter :: " << iter << " time :: " << TIMING_DELAY (tt1, tt2)/1000000/iter << " ticks :: " << TICK_DIFF (tt1, tt2) << std::endl;

  // Verification of the output
  bool isok = true;
  for (int i=0; i<n; ++i)
  {
    if (output[i] !=  op( input[i]))
    {
      std::cout << "Fail, i:" << i << ", @:" << output + i << ", input @:" << input + i << std::endl;
      isok = false;
    }
  }
  if (isok) std::cout << "Verification ok" << std::endl;

  delete [] input;
  delete [] output;

  return 0;
}
