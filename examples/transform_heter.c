#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "kaapi.h"


#define CONFIG_USE_STATIC 1


/* missing decls */
typedef struct kaapikaapi_processor_t kaapi_processor_t;
kaapi_processor_t* kaapi_stealcontext_kproc(kaapi_stealcontext_t*);
kaapi_processor_t* kaapi_request_kproc(kaapi_request_t*);
unsigned int kaapi_request_kid(kaapi_request_t*);
void kaapi_processor_set_workload(struct kaapi_processor_t*, kaapi_uint32_t);
void kaapi_processor_set_self_workload(kaapi_uint32_t);
unsigned int kaapi_processor_get_type(const kaapi_processor_t*); 


typedef struct range
{
  kaapi_access_t /* unsigned int* */ base;
  unsigned int i;
  unsigned int j;
} range_t;


static void create_range(range_t* range, unsigned int nelem)
{
  unsigned int i;
  unsigned int* base;

  base = malloc(nelem * sizeof(unsigned int));
  for (i = 0; i < nelem; ++i)
    base[i] = 0;
  *KAAPI_DATA(unsigned int*, range->base) = base;

  range->i = 0;
  range->j = nelem;
}


static void create_range2
(range_t* range, unsigned int* base, unsigned int i, unsigned j)
{
  *KAAPI_DATA(unsigned int*, range->base) = base;

  range->i = i;
  range->j = j;

  for (; i < j; ++i)
    base[i] = 0;
}


static unsigned int get_range_size(const range_t* range)
{
  return range->j - range->i;
}


static int steal_range
(range_t* sub, range_t* range, unsigned int size)
{
  if (get_range_size(range) < size)
    return -1;

  sub->j = range->j;
  sub->i = range->j - size;
  range->j = sub->i;

  return 0;
}


static int split_range
(range_t* sub, range_t* range, unsigned int size)
{
  if (get_range_size(range) == 0)
  {
    return -1;
  }
  else if (get_range_size(range) < size)
  {
    /* succeed even if size too large */
    size = get_range_size(range);
  }

  sub->i = range->i;
  sub->j = range->i + size;
  range->i += size;

  return 0;
}


typedef struct task_work
{
  volatile long lock;
  range_t range;
  kaapi_taskadaptive_result_t* ktr;
} task_work_t;


static task_work_t* alloc_work(kaapi_thread_t* thread)
{
  task_work_t* const work = kaapi_thread_pushdata_align
    (thread, sizeof(task_work_t), 8);

  kaapi_thread_allocateshareddata
    (&work->range.base, thread, sizeof(unsigned int*));

  work->range.i = 0;
  work->range.j = 0;

  work->lock = 0;
  work->ktr = NULL;

  return work;
}


static void lock_work(task_work_t* work)
{
  while (1)
  {
    if (__sync_bool_compare_and_swap(&work->lock, 0, 1))
      break;
  }
}


static void unlock_work(task_work_t* work)
{
  work->lock = 0;
  __sync_synchronize();
}


static int reduce_work(kaapi_stealcontext_t*, void*, void*, size_t, void*);

static void do_work(kaapi_stealcontext_t* sc, task_work_t* work)
{
  kaapi_taskadaptive_result_t* ktr;
  unsigned int i;
  int stealres;
  range_t subrange;

 compute_work:
  while (1)
  {
    lock_work(work);
    stealres = split_range(&subrange, &work->range, 512);
    *KAAPI_DATA(unsigned int*, subrange.base) =
      *KAAPI_DATA(unsigned int*, work->range.base);
    unlock_work(work);

    if (stealres == -1)
      break ;

    for (i = subrange.i; i < subrange.j; ++i)
      (*KAAPI_DATA(unsigned int*, subrange.base))[i] += 1;

    if (work->ktr != NULL)
    {
      if (kaapi_preemptpoint(work->ktr, sc, NULL, NULL, NULL, 0, NULL))
	return ;
    }
  }

  ktr = kaapi_get_thief_head(sc);
  if (ktr == NULL)
    return ;

  kaapi_preempt_thief(sc, ktr, NULL, reduce_work, work);
  goto compute_work;
}

static int split_work(kaapi_stealcontext_t*, int, kaapi_request_t*, void*);

static void common_entry(void* arg, kaapi_thread_t* thread)
{
  kaapi_stealcontext_t* const sc = kaapi_thread_pushstealcontext
    (thread, KAAPI_STEALCONTEXT_DEFAULT, split_work, arg, NULL);

  do_work(sc, arg);

  kaapi_steal_finalize(sc);
}


static void cuda_entry(void* arg, kaapi_thread_t* thread)
{
  range_t* const range = &((task_work_t*)arg)->range;
  printf("> cuda_entry [%u - %u[\n", range->i, range->j);
  common_entry(arg, thread);
  printf("< cuda_entry\n");
}


static void cpu_entry(void* arg, kaapi_thread_t* thread)
{
  range_t* const range = &((task_work_t*)arg)->range;
  printf("> cpu_entry [%u - %u[\n", range->i, range->j);
  common_entry(arg, thread);
  printf("< cpu_entry\n");
}


static int reduce_work
(kaapi_stealcontext_t* sc, void* targ, void* tptr, size_t tsize, void* vptr)
{
  task_work_t* const vwork = (task_work_t*)vptr;
  task_work_t* const twork = (task_work_t*)tptr;

  /* twork is a copy, dont lock */
  lock_work(vwork);
  vwork->range.i = twork->range.i;
  vwork->range.j = twork->range.j;
  unlock_work(vwork);

  return 0;
}


static int split_work
(kaapi_stealcontext_t* sc, int reqcount, kaapi_request_t* reqs, void* arg)
{
  task_work_t* const vwork = (task_work_t*)arg;
  task_work_t* twork;
  kaapi_thread_t* tthread;
  kaapi_task_t* ttask;
  kaapi_processor_t* tproc;
  unsigned int rangesize;
  unsigned int unitsize;
  range_t subrange;
  int stealres = -1;
  int repcount = 0;

  lock_work(vwork);

  rangesize = get_range_size(&vwork->range);

#if 0 /* fixme */
  if ((int)rangesize < 0)
  {
    unlock_work(vwork);
    return 0;
  }
#endif /* fixme */

  if (rangesize > 512)
  {
    unitsize = rangesize / (reqcount + 1);
    if (unitsize == 0)
    {
      unitsize = 512;
      reqcount = rangesize / 512;
    }

    stealres = steal_range
      (&subrange, &vwork->range, unitsize * reqcount);
  }
  unlock_work(vwork);

  if (stealres == -1)
    return 0;

  for (; reqcount > 0; ++reqs)
  {
    if (!kaapi_request_ok(reqs))
      continue ;

    tthread = kaapi_request_getthread(reqs);
    ttask = kaapi_thread_toptask(tthread);

    tproc = kaapi_request_kproc(reqs);

    if (kaapi_processor_get_type(tproc) == KAAPI_PROC_TYPE_CPU)
      kaapi_task_init(ttask, cpu_entry, NULL);
    else
      kaapi_task_init(ttask, cuda_entry, NULL);

    twork = alloc_work(tthread);
    *KAAPI_DATA(unsigned int*, twork->range.base) =
      *KAAPI_DATA(unsigned int*, vwork->range.base);
    twork->ktr = kaapi_allocate_thief_result(sc, sizeof(task_work_t), NULL);

    split_range(&twork->range, &subrange, unitsize);

    kaapi_task_setargs(ttask, (void*)twork);
    kaapi_thread_pushtask(tthread);
    kaapi_request_reply_head(sc, reqs, twork->ktr);

    --reqcount;
    ++repcount;
  }

  return repcount;
}


static void register_task_format(void)
{
#define PARAM_COUNT 3

#define kaapi_mem_format kaapi_long_format

  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_RW, KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_R };

  static const kaapi_offset_t offsets[PARAM_COUNT] =
    { offsetof(task_work_t, range.base),
      offsetof(task_work_t, range.i),
      offsetof(task_work_t, range.j) };

  const struct kaapi_format_t* formats[PARAM_COUNT] =
    { kaapi_mem_format, kaapi_uint_format, kaapi_uint_format };

  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister
    (format, NULL, "heter_task", sizeof(task_work_t),
     PARAM_COUNT, modes, offsets, formats);

  kaapi_format_taskregister_body(format, cpu_entry, KAAPI_PROC_TYPE_CPU);
  kaapi_format_taskregister_body(format, cuda_entry, KAAPI_PROC_TYPE_CUDA);
}


static void __attribute__((unused))
main_adaptive_entry(unsigned int nelem)
{
  task_work_t* work;
  kaapi_thread_t* thread;
  kaapi_task_t* task;
  kaapi_frame_t frame;

  register_task_format();

  thread = kaapi_self_thread();
  kaapi_thread_save_frame(thread, &frame);

  /* processing task */
  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, cpu_entry, NULL);

  work = alloc_work(thread);
  create_range(&work->range, nelem);

  kaapi_task_setargs(task, (void*)work);
  kaapi_thread_pushtask(thread);

#if 0
  /* checking task */
  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, print_entry, (void*)work);
  kaapi_thread_pushtask(thread);
#endif

  kaapi_sched_sync();
  kaapi_thread_restore_frame(thread, &frame);
}


static void __attribute__((unused))
main_static_entry(unsigned int nelem)
{
  task_work_t* work;
  kaapi_task_t* task;
  kaapi_threadgroup_t group;

  register_task_format();

#define PARTITION_COUNT 2
#define PARTITION_ID_CPU 0
#define PARTITION_ID_GPU 1

  kaapi_threadgroup_create(&group, PARTITION_COUNT);

  kaapi_threadgroup_begin_partition(group);

  /* cpu partition */
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU);
  kaapi_task_init(task, cpu_entry, NULL);
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU));
  create_range(&work->range, nelem);
  work->range.j = nelem / 2; /* split the work */
  kaapi_task_setargs(task, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU);

  /* gpu partition */
  unsigned int* base = *KAAPI_DATA(unsigned int*, work->range.base);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU);
  kaapi_task_init(task, cuda_entry, NULL);
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU));
  create_range2(&work->range, base, nelem / 2, nelem);
  kaapi_task_setargs(task, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU);

  kaapi_threadgroup_end_partition(group);

  /* execute */
  kaapi_threadgroup_begin_execute(group);
  kaapi_threadgroup_end_execute(group);

  kaapi_threadgroup_destroy(group);
}


int main(int ac, char** av)
{
#if CONFIG_USE_STATIC
  main_static_entry(100000);
#else
  main_adaptive_entry(100000);
#endif
  return 0;
}
