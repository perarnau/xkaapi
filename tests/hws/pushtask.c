#include "kaapi.h"


int main(int ac, char** av)
{
  static void* numa_task = (void*)42;
  static void* numa_data = (void*)42;

  static void* flat_task = (void*)24;
  static void* flat_data = (void*)24;

  kaapi_init(1, &ac, &av);

  kaapi_hws_pushtask_flat(flat_task, flat_data);
  kaapi_hws_pushtask_numa(numa_task, numa_data);
  while (1) ;

  kaapi_finalize();

  return 0;
}
