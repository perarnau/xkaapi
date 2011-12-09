/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 ** 
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 ** 
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 ** 
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 ** 
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 ** 
 */
#include "kaapic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>


/** Description of the example.
 
 Overview of the execution.
 
 What is shown in this example.
 
 Next example(s) to read.
 */


static void apply_cos(
  int32_t i, int32_t j, int32_t tid, double* array
)
{
  /* process array[i, j[, inclusive */
  int32_t k;

  for (k = i; k < j; ++k)
    array[k] += cos(array[k]);
}


/**
 */
int main(int ac, char** av)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;
  
#define ITEM_COUNT 100000
  static double array[ITEM_COUNT];
  
  /* initialize the runtime */
  kaapic_init(1);
  
  for (iter = 0; iter < 100; ++iter)
  {
    /* initialize, apply, check */
    for (i = 0; i < ITEM_COUNT; ++i)
      array[i] = 0.f;

    t0 = kaapic_get_time();
    kaapic_foreach( 0, ITEM_COUNT, 0, 1, apply_cos, array );
    t1 = kaapic_get_time();
    sum += (t1-t0)*1000; /* ms */

    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] != 1.f)
      {
        printf("invalid @%lu == %lf\n", i, array[i]);
        break ;
      }
  }

  printf("done: %lf (ms)\n", sum / 100);

  /* finalize the runtime */
  kaapic_finalize();
  
  return 0;
}
