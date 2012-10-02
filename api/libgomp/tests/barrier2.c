#include <stdio.h>
#include <omp.h>

#include "test-toolbox.h"

int
main (int argc, char **argv)
{
  int cpt = 0, total = 0;
  int nthreads = 128;

#pragma omp parallel shared (cpt, total) num_threads (nthreads)
  {
#pragma omp critical
    cpt++;

#pragma omp barrier

#pragma omp critical
    total += cpt;
  }

  return !(total == cpt * nthreads);
}
