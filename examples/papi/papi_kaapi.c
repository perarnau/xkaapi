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

/* see README for info related to the papi usage in kaapi
   see examples/for_each for general code information
 */

#include "kaapic.h"
#include <string.h>
#include <math.h>


/**
 */
static void apply_cos( int i, int j, int tid, double* array )
{
  for (int k=i; k<j; ++k)
    array[k] += cos(array[k]);
}


/**
 */
int main(int argc, char** argv)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;
  size_t size;

  /* at most 3 papi counters */
  kaapi_perf_idset_t perfids;
  kaapi_perf_counter_t counters[3] = {0, 0, 0};
  
  if (argc>1)
    size = atoi(argv[1]);
  else
    size = 100000;

  double* array = (double*)malloc(sizeof(double)*size);
  
  /* initialize the runtime */
  kaapic_init(KAAPIC_START_ONLY_MAIN);

  /* create performance counter id set. use the
     KAAPI_PERF_PAPIES=a,b,c environ variable to
     bind KAAPI_PERF_ID_PAPI_N. man papi_avail(3)
     to see how to list available counters.
   */
  kaapi_perf_idset_zero(&perfids);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_0);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_1);
  kaapi_perf_idset_add(&perfids, KAAPI_PERF_ID_PAPI_2);
  
  for (iter = 0; iter < 100; ++iter)
  {
    /* initialize, apply, check */
    for (i = 0; i < size; ++i)
      array[i] = 0.f;

    t0 = kaapi_get_elapsedns();
    kaapic_foreach( 0, size, 0, 1, apply_cos, array );
    t1 = kaapi_get_elapsedns();
    sum += (t1-t0)/1000; /* ms */

    for (i = 0; i < size; ++i)
      if (array[i] != 1.f)
      {
        printf("invalid @%lu == %lf\n", i, array[i]);
        break ;
      }
  }

  printf("done: %lf (ms)\n", sum / 100);

  /* accumulate all the processors counters and report
     counters[i] where i in KAAPI_PERF_PAPIES[i]
   */
  kaapi_perf_accum_counters(&perfids, counters);
  printf("perf_counters: %lu, %lu, %lu\n",
	 counters[0], counters[1], counters[2]);

  /* finalize the runtime */
  kaapic_finalize();
  
  free(array);

  return 0;
}
