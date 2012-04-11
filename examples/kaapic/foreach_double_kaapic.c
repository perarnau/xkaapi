#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "kaapic.h"



static void fu1 (
  int32_t i, int32_t j, int32_t tid, 
  double* array
)
{
  /* process array[i, j], inclusive */
  int32_t k;  

  for (k = i; k < j; ++k)
  {
    array[k] = sin(sin(array[k]));
  }  
}

static void fu0 (
  int32_t i, int32_t j, int32_t tid, 
  double* array, double* value
)
{
  kaapic_foreach(i, j, 0, 1, fu1, array );
}


int main(int ac, char** av)
{
  static double array[100000 * 48];
  static const int32_t size = 100000 * 48;
  static const double one = 1;
  int32_t i;
  kaapic_foreach_attr_t attr;
  
  for (i = 0; i < size; ++i) array[i] = 0;
  
  kaapic_init(1);
  
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, 256, 1024);
  
  for (i = 0; i < 10000; ++i)
  {
#if 1
    if (i % 10 ==0){fputc('.',stdout);fflush(stdout);}
    if ((1+i) % 1000 ==0) printf("\n%i\n", i);
#endif
    kaapic_foreach(0, size, &attr, 2, fu0, array, &one);      
  }

  kaapic_finalize();

  return 0;
}
