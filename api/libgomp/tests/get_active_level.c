#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

/* OpenMP31 Spec: 3.2.19: The omp_get_active_level routine returns the
 * number of nested, active parallel regions enclosing the task that
 * contains the call. The routine always returns a non-negative
 * integer, and returns 0 if it is called from the sequential part of
 * the program. */

int
main (int argc, char **argv)
{
  int ret = 0;

  if (omp_get_active_level () != 0)
    return 1;
  
#pragma omp parallel num_threads (2) shared (ret)
  {
    if (omp_get_active_level () != 1)
      ret = 2;
  }

#pragma omp parallel if (0) num_threads (2) shared (ret)
  {
    if (omp_get_active_level () != 0)
      ret = 3;
  }

  omp_set_nested (1);

#pragma omp parallel num_threads (2) shared (ret)
  {
#pragma omp parallel num_threads (2) shared (ret)
    if (omp_get_active_level () != 2)
      ret = 4;
  }
 
 return ret;
}
