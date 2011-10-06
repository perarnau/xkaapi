#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include "kaapi.h"


typedef struct wrapped_uint
{
  kaapi_access_t val;
} wrapped_uint_t;


static void flat_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);
  printf("[%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);
  usleep(100000);
}


static void numa_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;
  const unsigned int val = *kaapi_data(unsigned int, &wui->val);
  printf("[%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);
  usleep(100000);
}


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


int main(int ac, char** av)
{
  unsigned int j;

  kaapi_init(1, &ac, &av);

  for (j = 0; j < 4; ++j)
  {
    kaapi_thread_t* const thread = kaapi_self_thread();
    kaapi_task_t* task;
    wrapped_uint_t* wui;

    /* push for flat stealing */
    task = kaapi_thread_toptask(thread);
    wui = kaapi_thread_pushdata_align
      (thread, sizeof(wrapped_uint_t), sizeof(void*));
    kaapi_thread_allocateshareddata
      (&wui->val, thread, sizeof(unsigned int));
    *kaapi_data(unsigned int, &wui->val) = j;
    kaapi_task_initdfg(task, flat_body, wui);
    kaapi_thread_pushtask(thread);

    /* push for numa stealing */

    /* todo: should be an allocator per ws_block, ie.
       ws->allocate(). export hws_alloc on top of that.
     */

    wui = kaapi_thread_pushdata_align
      (thread, sizeof(wrapped_uint_t), sizeof(void*));
    kaapi_thread_allocateshareddata
      (&wui->val, thread, sizeof(unsigned int));
    *kaapi_data(unsigned int, &wui->val) = j;
    task = kaapi_thread_toptask(thread);
    kaapi_task_initdfg(task, numa_body, wui);
    kaapi_thread_distribute_task(thread, KAAPI_HWS_LEVELID_NUMA);
    usleep(100000);
  }

  kaapi_sched_sync();

  printf("done\n");

  kaapi_finalize();

  return 0;
}
