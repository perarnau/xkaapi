#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include "kaapi.h"


typedef struct wrapped_uint
{
  kaapi_access_t val;
  unsigned int stor;
} wrapped_uint_t;


static void flat_body(void* p, kaapi_thread_t* t)
{
  wrapped_uint_t* wui = (wrapped_uint_t*)p;

  const unsigned int val = *kaapi_data(unsigned int, &wui->val);

  printf("xx [%u] %s %u\n", kaapi_get_self_kid(), __FUNCTION__, val);
  usleep(1000000);

  *kaapi_data(unsigned int, &wui->val) = 3;
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


int main(int ac, char** av)
{
  wrapped_uint_t flat_wui;
  unsigned int j;

  kaapi_init(1, &ac, &av);

  for (j = 0; j < 128; ++j)
  {
    kaapi_thread_t* const thread = kaapi_self_thread();
    kaapi_task_t* const task = kaapi_thread_toptask(thread);

    flat_wui.stor = 42;
    kaapi_access_init(&flat_wui.val, &flat_wui.stor);
    kaapi_task_initdfg(task, flat_body, &flat_wui);
    kaapi_thread_pushtask(thread);
  }

  kaapi_sched_sync();

  printf("done: %u\n", flat_wui.stor);

  kaapi_finalize();

  return 0;
}
