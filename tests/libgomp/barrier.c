#include <stdio.h>
#include <omp.h>

#include "test-toolbox.h"

int
main (int argc, char **argv)
{
  int cpt = 0, total = 0;
  int nthreads = 0;

#pragma omp parallel
  {
    nthreads = omp_get_num_threads ();
  }

#pragma omp parallel shared (cpt, total)
  {
#pragma omp critical
    cpt++;

#pragma omp barrier

#pragma omp critical
    total += cpt;
  }

  return !(total == cpt * nthreads);
}
