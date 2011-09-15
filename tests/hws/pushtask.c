#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "kaapi.h"


typedef struct wrapped_uint
{
  kaapi_access_t val;
} wrapped_uint_t;


static void numa_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);
  printf("[%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);
}


static void flat_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);
  printf("[%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);
}


KAAPI_REGISTER_TASKFORMAT
(
 numa_format,
 "numa",
 numa_body,
 sizeof(wrapped_uint_t),
 1,
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_R },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.data) },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.version) },
 (const struct kaapi_format_t*[]){ kaapi_int_format },
 (struct kaapi_memory_view_t[]){},
 0
)


KAAPI_REGISTER_TASKFORMAT
(
 flat_format,
 "flat",
 flat_body,
 sizeof(wrapped_uint_t),
 1,
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_R },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.data) },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.version) },
 (const struct kaapi_format_t*[]){ kaapi_int_format },
 (struct kaapi_memory_view_t[]){},
 0
)


int main(int ac, char** av)
{
  kaapi_init(1, &ac, &av);

  kaapi_hws_pushtask_flat(flat_body, (void*)(uintptr_t)42);
  kaapi_hws_pushtask_numa(numa_body, (void*)(uintptr_t)24);
  kaapi_sched_sync();
  while (1) ;

  kaapi_finalize();

  return 0;
}
