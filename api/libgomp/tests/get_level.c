#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

/* OpenMP31 Spec: 3.2.16: The omp_get_level routine returns the number
 * of nested parallel regions (whether active or inactive) enclosing
 * the task that contains the call, not including the implicit
 * parallel region. The routine always returns a non-negative integer,
 * and returns 0 if it is called from the sequential part of the
 * program. */

int
main (int argc, char **argv)
{
  int ret = 0;

  if (omp_get_level () != 0)
    return 1;
  
#pragma omp parallel num_threads (2) shared (ret)
  {
    if (omp_get_level () != 1)
      ret = 2;
  }

#pragma omp parallel if (0) num_threads (2) shared (ret)
  {
    if (omp_get_level () != 1)
      ret = 3;
  }

  omp_set_nested (1);

#pragma omp parallel num_threads (2) shared (ret)
  {
#pragma omp parallel num_threads (2) shared (ret)
    if (omp_get_level () != 2)
      ret = 4;
  }
 
 return ret;
}
