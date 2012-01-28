#include <stdio.h>

#include "test-toolbox.h"

#define OK 1
#define KO 0

int
main (int argc, char **argv)
{
  int res[8] = { 0 };
#pragma omp parallel shared (res)
  {
    int i;

    for (i = 0; i < 8; i++)
      {
#pragma omp task firstprivate (i) shared (res)
	{
	  res[i]++;
	}
      }
  }

  int i, passed = OK;
  for (i = 0; i < 8; i++)
    {
      if (res[i] != 1)
	passed = KO;
    }
  
  test_check ("task-firstprivate", (passed == OK));
  
  return 0;
}
