#include "kaapi.h"


static void numa_task(void* p)
{
  printf("[%u] %s\n", kaapi_get_self_kid(), __FUNCTION__);  
}


static void flat_task(void* p)
{
  printf("[%u] %s\n", kaapi_get_self_kid(), __FUNCTION__);
}


int main(int ac, char** av)
{
  kaapi_init(1, &ac, &av);

  kaapi_hws_pushtask_flat(flat_task, NULL);
  kaapi_hws_pushtask_numa(numa_task, NULL);
  kaapi_sched_sync();
  while (1) ;

  kaapi_finalize();

  return 0;
}
