#include <stdio.h>
#include <stdint.h>
#include "kaapi.h"


static void numa_task(void* p, kaapi_thread_t* t)
{
  printf("[%u] %s %lx\n", kaapi_get_self_kid(), __FUNCTION__, (uintptr_t)p);
}


static void flat_task(void* p, kaapi_thread_t* t)
{
  printf("[%u] %s %lx\n", kaapi_get_self_kid(), __FUNCTION__, (uintptr_t)p);
}


int main(int ac, char** av)
{
  kaapi_init(1, &ac, &av);

  kaapi_hws_pushtask_flat(flat_task, (void*)(uintptr_t)42);
  kaapi_hws_pushtask_numa(numa_task, (void*)(uintptr_t)24);
  kaapi_sched_sync();
  while (1) ;

  kaapi_finalize();

  return 0;
}
