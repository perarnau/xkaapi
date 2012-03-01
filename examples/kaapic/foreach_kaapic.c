#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "kaapic.h"


/* some notes
 . in entrypoint, by value params must be passed as pointers
 . depends on kaapif library
 . intervals are inclusive
 . the entrypoint prototype takes 3 first args as pointers
 */


#define CONFIG_TERM_BUG 0
#define CONFIG_CHECK_RES 0
#define CONFIG_BENCH 1


#if CONFIG_TERM_BUG
static void* volatile global_check __attribute__((aligned));
#endif


static void fu (
                int32_t i, int32_t j, int32_t tid, double* array, double* value
#if CONFIG_TERM_BUG
                , void* check
#endif
                )
{
  /* process array[i, j], inclusive */
  
  int32_t k;
  
  for (k = i; k < j; ++k)
  {
#if CONFIG_TERM_BUG
    if (array[k] != (double)(uintptr_t)check)
    {
      printf("INVALID_ARRAY (%lf != %lf)\n",
             array[k], (double)(uintptr_t)check);
      fflush(stdout);
      while (1) ;
    }
#endif
    
    array[k] += *value;
    array[k] += sqrt(sin(*value * *value) + cos(*value * *value));
    array[k] += sin(*value * *value) + sqrt(cos(*value * *value));
  }
  
#if CONFIG_TERM_BUG
  if (global_check != check)
  {
    printf("INVALID_GLOBAL_CHECK (%p != %p)\n",
           global_check, check);
    fflush(stdout);
    while (1) ;
  }
#endif
}


int main(int ac, char** av)
{
  static double array[100000 * 48];
  static const int32_t size = 100000 * 48;
  static const double one = 1;
  int32_t i;
  kaapic_foreach_attr_t attr;
  
#if CONFIG_BENCH
  double start, stop;
#endif
  
  for (i = 0; i < size; ++i) array[i] = 0;
  
  kaapic_init(1);
  
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, 1024, 1024);
  
#if CONFIG_BENCH
  start = kaapic_get_time();
#endif
  
#if CONFIG_TERM_BUG
  for (i = 0; i < 10000; ++i)
#else
    for (i = 0; i < 1000; ++i)
#endif
    {
#if 0
     if (i % 10 ==0) fputc('.',stdout);
     if (i % 100 ==0) fflush(stdout);
     if (i % 1000 ==0) printf("%i\n", i);
#endif
#if CONFIG_TERM_BUG
      global_check = (void*)(uintptr_t)i;
      //OLD    kaapic_foreach(fu, 0, size - 1, 3, array, &one, global_check);
      kaapic_foreach(0, size, &attr, 3, fu, array, &one, global_check);
      
#else
      //OLD    kaapic_foreach(fu, 0, size - 1, 2, array, &one);
      kaapic_foreach(0, size, &attr, 2, fu, array, &one);
#endif
      
      /* check the result after each iteration. disable when timing. */
#if CONFIG_CHECK_RES
      const double refval = (double)(i + 1);
      unsigned int j;
      for (j = 0; j < size; ++j)
      {
        if (array[j] != refval)
        {
          printf("invalid @%d: %lf != %lf\n", j, array[j], refval);
          fflush(stdout);
          array[j] = refval;
          while (1) ;
          break ;
        }
      }
#endif
    }
  
#if CONFIG_BENCH
  stop = kaapic_get_time();
#endif
  
  
#if CONFIG_BENCH
  printf("time: %lf\n", stop - start);
#endif
  
  kaapic_finalize();

  return 0;
}
