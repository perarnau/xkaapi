#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

int
main (int argc, char **argv)
{
  if (omp_get_ancestor_thread_num (0) != 0)
    return 1;

  if (omp_get_ancestor_thread_num (1) != -1)
    return 2;

  int ret = 0;

#pragma omp parallel num_threads (2) shared (ret)
  {
    int tid = omp_get_thread_num ();
        
    if (omp_get_ancestor_thread_num (0) != 0)
      ret = 3;    

    if (omp_get_ancestor_thread_num (1) != tid)
      ret = 4;
  }

  omp_set_nested (1);

#pragma omp parallel num_threads (4) shared (ret)
  {
    int tid = omp_get_thread_num ();
#pragma omp parallel num_threads (3) firstprivate (tid) shared (ret)
    {
      int tid2 = omp_get_thread_num ();

      if (omp_get_ancestor_thread_num (1) != tid)
	ret = 5;

      if (omp_get_ancestor_thread_num (2) != tid2)
	ret = 6;
    }
  }

  return ret;
}
