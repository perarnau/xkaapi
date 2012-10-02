#include <stdio.h>

#include "test-toolbox.h"

#define OK 1
#define KO 0

int
main (int argc, char **argv)
{
  int res[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

#pragma omp parallel shared (res)
  {
#pragma omp single 
    {
      int i;
      
      for (i = 0; i < 8; i++)
      {
#pragma omp task firstprivate (i) shared (res)
        res[i]++;
      }
    }
#pragma omp taskwait
  }
  
  int i, passed = OK;
  for (i = 0; i < 8; i++)
  {
    if (res[i] != 1)
      passed = KO;
  }
  
  return !(passed == OK);
}
