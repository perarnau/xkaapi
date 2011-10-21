#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <string.h>
#include <numa.h>
#include "kaapi.h"


#define CONFIG_USE_NUMA_HWS 1


typedef struct range
{
  unsigned int* p;
  unsigned int i;
  unsigned int j;
} range_t;


static void addone_body(void* fu, kaapi_thread_t* t)
{
  range_t* r = (range_t*)fu;

  unsigned int* p = r->p;
  unsigned int i = r->i;
  unsigned int j = r->j;

#if CONFIG_NUMA_HWS /* check mapping */
  {
    const unsigned int id = kaapi_get_self_cpu_id();
    const unsigned int nodeid = numa_node_of_cpu((int)id);
#if 0 /* not working on cpu != 0 */
    const unsigned int pageid = kaapi_numa_get_page_node(p);
#else
    const unsigned int pageid = ((i * sizeof(unsigned int)) / 0x1000) % 8;
#endif
    if (pageid != nodeid)
    {
      printf("invalid mapping: page %lx @ %u\n", (unsigned long)(p + i), pageid);
      while (1) ;
    }
  }
#endif

  for (; i < j; ++i) ++p[i];
}


KAAPI_REGISTER_TASKFORMAT
(
 addone_format,
 "addone",
 addone_body,
 sizeof(range_t),
 3,
 (kaapi_access_mode_t[]){ KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V },
 (kaapi_offset_t[]){ offsetof(range_t, p), offsetof(range_t, i), offsetof(range_t, j) },
 (kaapi_offset_t[]){ 0, 0, 0 },
 (const struct kaapi_format_t*[]){ kaapi_ulong_format, kaapi_int_format, kaapi_int_format },
 (struct kaapi_memory_view_t[]){},
 0
)


static void* allocate_array(unsigned int size)
{
  static const size_t page_size = 0x1000;

#if CONFIG_USE_NUMA_HWS
  const unsigned int node_count =
    kaapi_hws_get_node_count(KAAPI_HWS_LEVELID_NUMA);
#else
  const unsigned int node_count = 8;
#endif

  const unsigned int page_count = size / page_size;

  unsigned int i;
  void* p;

  /* allocate */
  if (posix_memalign(&p, page_size, size))
  {
    perror("posix_memalign");
    exit(-1);
  }

  /* bind the pages */
  for (i = 0; i < page_count; ++i)
  {
    const void* const addr = (const void*)((uintptr_t)p + i * page_size);
    if (kaapi_numa_bind(addr, page_size, i % node_count))
    {
      printf("[!] kaapi_numa_bind\n");
      exit(-1);
    }

    if (kaapi_numa_get_page_node(addr) != (i % node_count))
    {
      printf("ivnalid_barfu\n");
      exit(-1);
    }
  }

  return p;
}


int main(int ac, char** av)
{
  static const unsigned int size = 4 * 1024 * 1024;

  unsigned int* array;
  unsigned int npages;
  unsigned int i;
  double fu, bar;
#if CONFIG_USE_NUMA_HWS
  unsigned int node_count;
#endif

  kaapi_init(1, &ac, &av);

#if CONFIG_USE_NUMA_HWS
  node_count = kaapi_hws_get_node_count(KAAPI_HWS_LEVELID_NUMA);
#endif

  array = allocate_array(size);
  memset(array, 0, size);

  npages = size / 0x1000;

  fu = kaapi_get_elapsedns();

  for (i = 0; i < npages; ++i)
  {
    kaapi_thread_t* const thread = kaapi_self_thread();
    kaapi_task_t* task;
    range_t* range;

    range = kaapi_thread_pushdata_align
      (thread, sizeof(range_t), sizeof(void*));
    range->p = array;
    range->i = (i * 0x1000) / sizeof(unsigned int);
    range->j = range->i + (0x1000 / sizeof(unsigned int));

    task = kaapi_thread_toptask(thread);
    kaapi_task_initdfg(task, addone_body, range);

#if CONFIG_USE_NUMA_HWS
    kaapi_thread_distribute_task_with_nodeid
      (thread, KAAPI_HWS_LEVELID_NUMA, i % node_count);
#else
    kaapi_thread_pushtask(thread);
#endif
  }

  kaapi_sched_sync();

  bar = kaapi_get_elapsedns();

  printf("done: %lf us.\n", (bar - fu) / 1000);

  kaapi_finalize();

  for (i = 0; i < size / sizeof(unsigned int); ++i)
    if (array[i] != 1)
    {
      printf("error@%u == %u\n", i, array[i]);
      break ;
    }

  free(array);

  return 0;
}
