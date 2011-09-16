#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "kaapi.h"


typedef struct wrapped_uint
{
  kaapi_access_t val;
  unsigned int stor;
} wrapped_uint_t;


static void numa_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);

  printf("xx [%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);

  *kaapi_data(unsigned int, &wui->val) = 3;
}


static void flat_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);

  printf("xx [%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);

  *kaapi_data(unsigned int, &wui->val) = 3;
}


KAAPI_REGISTER_TASKFORMAT
(
 numa_format,
 "numa",
 numa_body,
 sizeof(wrapped_uint_t),
 1,
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_RW },
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
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_RW },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.data) },
 (kaapi_offset_t[]){ offsetof(wrapped_uint_t, val.version) },
 (const struct kaapi_format_t*[]){ kaapi_int_format },
 (struct kaapi_memory_view_t[]){},
 0
)


int main(int ac, char** av)
{
  wrapped_uint_t numa_wui;
  wrapped_uint_t flat_wui;

  kaapi_init(1, &ac, &av);

  {
    unsigned int j;
    for (j = 0; j < 128; ++j)
    {
      /* flat_wui.stor = 42; */
      /* kaapi_access_init(&flat_wui.val, &flat_wui.stor); */
      /* kaapi_hws_pushtask_flat(flat_body, (void*)&flat_wui); */

      numa_wui.stor = 24;
      kaapi_access_init(&numa_wui.val, &numa_wui.stor);
      kaapi_hws_pushtask_numa(numa_body, (void*)&numa_wui);
    }
  }

  kaapi_hws_sched_sync();

  printf("done: %u %u\n", flat_wui.stor, numa_wui.stor);

  kaapi_hws_print_counters();

  kaapi_finalize();

  return 0;
}
