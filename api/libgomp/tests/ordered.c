#include <stdio.h>
#include <string.h>
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
  
  /* Generate the string to compare the output with. */
  char magic[nthreads + 1];
  char outstring[nthreads + 1];
  
  for (int i = 0; i < nthreads; i++)
    sprintf (magic + i, "%i", i);
  magic[nthreads] = '\0';

  int cur = 0;
  
#pragma omp parallel shared (outstring, cur)
  {
    int i = 0;
    
#pragma omp for schedule (static) ordered
    for (i = 0; i < nthreads; i++)
    {
#pragma omp ordered
      {
        sprintf (outstring + cur, "%i", i);
        cur++;
      }
    }
  }
  
  outstring[nthreads] = '\0';
  
  test_check ("ordered", (strcmp (outstring, magic) == 0));
  
  return 0;
}