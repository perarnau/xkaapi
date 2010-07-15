#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "kaapi.h"
#include "kaapi_cuda_func.h"


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


static void create_range
(range_t* range, unsigned int* base, unsigned int i, unsigned j)
{
  kaapi_access_init(&range->base, (void*)base);
  range->i = i;
  range->j = j;
}


static inline unsigned int get_range_size(const range_t* range)
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

  work->range.base.data = NULL;
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
  unsigned int* const base = (unsigned int*)work->range.base.data;

  kaapi_taskadaptive_result_t* ktr;
  unsigned int i;
  int stealres;
  range_t subrange;

 compute_work:
  while (1)
  {
    lock_work(work);
    stealres = split_range(&subrange, &work->range, 512);
    unlock_work(work);

    if (stealres == -1)
      break ;

    for (i = subrange.i; i < subrange.j; ++i)
      base[i] += 1;

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


/* add1 task */

#if 0 /* prepost scheme */

static void cuda_pre_handler
(void* arg, kaapi_thread_t* thread, kaapi_cuda_kernel_dim_t* dim)
{
  range_t* const range = &((task_work_t*)arg)->range;

  printf("> cuda_pre_handler [%u - %u[\n", range->i, range->j);

  dim->x = 256;
  dim->y = 1;
  dim->z = 1;
}

static void cuda_post_handler
(void* arg, kaapi_thread_t* thread, int error)
{
  printf("> cuda_post_handler (%d)\n", error);
}

#elif 1 /* high level call */

#include <stdint.h>
typedef uintptr_t kaapi_mem_addr_t;
extern void kaapi_mem_synchronize(kaapi_mem_addr_t, size_t);
extern void kaapi_mem_synchronize2(kaapi_mem_addr_t, size_t);

static void add1_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> add1_cuda_entry [%u - %u[\n", range->i, range->j);

#if 1 /* driver api */
  {
#define THREAD_DIM_X 256
    static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
    static const kaapi_cuda_dim2_t bdim = {1, 1};

    kaapi_cuda_func_t fn;

    kaapi_cuda_func_init(&fn);

#define STERN "transform_heter"
    kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "add1");

    kaapi_cuda_func_push_ptr(&fn, base);
    kaapi_cuda_func_push_uint(&fn, range->i);
    kaapi_cuda_func_push_uint(&fn, range->j);

    kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

    kaapi_cuda_func_unload_ptx(&fn);
  }
#else /* c++ api */
  {
    add1_heter<<<1, 256, 0, stream>>>
      (base, range->i, range->j);
  }
#endif /* driver api */

  printf("< add1_cuda_entry\n");
}

#else /* cpu call */

static void cuda_entry(void* arg, kaapi_thread_t* thread)
{
  range_t* const range = &((task_work_t*)arg)->range;
  printf("> cuda_entry [%u - %u[\n", range->i, range->j);
  common_entry(arg, thread);
  printf("< cuda_entry\n");
}

#endif /* prepost scheme */


/* add1 task */

static void add1_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  range_t* const range = &((task_work_t*)arg)->range;
  printf("> add1_cpu_entry [%u - %u[\n", range->i, range->j);
  common_entry(arg, thread);
  printf("< add1_cpu_entry\n");
}


/* mul2 task */

static void mul2_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> mul2_cuda_entry [%u - %u[\n", range->i, range->j);

  static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
  static const kaapi_cuda_dim2_t bdim = {1, 1};

  kaapi_cuda_func_t fn;

  kaapi_cuda_func_init(&fn);

  kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "mul2");

  kaapi_cuda_func_push_ptr(&fn, base);
  kaapi_cuda_func_push_uint(&fn, range->i);
  kaapi_cuda_func_push_uint(&fn, range->j);

  kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

  kaapi_cuda_func_unload_ptx(&fn);

  printf("< mul2_cuda_entry\n");
}

static void mul2_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  unsigned int* const base = range->base.data;

  printf("> mul2_cpu_entry [%u - %u[\n", range->i, range->j);

  unsigned int i;
  for (i = range->i; i < range->j; ++i)
    base[i] *= 2;

  printf("< mul2_cpu_entry\n");
}


/* adaptive */

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
      kaapi_task_init(ttask, add1_cpu_entry, NULL);
    else
      kaapi_task_init(ttask, (kaapi_task_body_t)add1_cuda_entry, NULL);

    twork = alloc_work(tthread);
    kaapi_access_init(&twork->range.base, vwork->range.base.data);
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


/* common for every tasks working with task_work_t */

#define PARAM_COUNT 3

#define kaapi_mem_format kaapi_ulong_format

static const kaapi_offset_t offsets[PARAM_COUNT] =
{
  offsetof(task_work_t, range.base),
  offsetof(task_work_t, range.i),
  offsetof(task_work_t, range.j)
};

static size_t get_param_size
(const struct kaapi_format_t* format, unsigned int i, const void* data)
{
  /* assume i == 0 */
  const task_work_t* const work = (task_work_t*)data;
  return (size_t)get_range_size(&work->range) * sizeof(unsigned int);
}

static void register_add1_task_format(void)
{
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
  { KAAPI_ACCESS_MODE_RW, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };

  const struct kaapi_format_t* formats[PARAM_COUNT] =
  { kaapi_mem_format, kaapi_uint_format, kaapi_uint_format };

  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister
    (format, NULL, "add1_task", sizeof(task_work_t),
     PARAM_COUNT, modes, offsets, formats, get_param_size);

  kaapi_format_taskregister_body
    (format, add1_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)add1_cuda_entry, KAAPI_PROC_TYPE_CUDA);
}

static void register_mul2_task_format(void)
{
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
  { KAAPI_ACCESS_MODE_RW, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };

  const struct kaapi_format_t* formats[PARAM_COUNT] =
  { kaapi_mem_format, kaapi_uint_format, kaapi_uint_format };

  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister
    (format, NULL, "mul2_task", sizeof(task_work_t),
     PARAM_COUNT, modes, offsets, formats, get_param_size);

  kaapi_format_taskregister_body
    (format, mul2_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)mul2_cuda_entry, KAAPI_PROC_TYPE_CUDA);
}

/* memset task */

static void memset_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> memset_cuda_entry [%u - %u[\n", range->i, range->j);

#define MEMSET_VALUE 1
  CUresult res = cuMemsetD32(base, MEMSET_VALUE, get_range_size(range));
  if (res != CUDA_SUCCESS)
    printf("cudaError %u\n", res);

  printf("< memset_cuda_entry\n");
}

static void memset_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  unsigned int* const base = (unsigned int*)range->base.data;
  unsigned int i;

  printf("> memset_cpu_entry [%u - %u[\n", range->i, range->j);

  for (i = 0; i <  range->j; ++i)
    base[i] = MEMSET_VALUE;

  printf("< memset_cpu_entry\n");
}

static void register_memset_task_format(void)
{
  struct kaapi_format_t* const format = kaapi_format_allocate();

  static const kaapi_access_mode_t modes[PARAM_COUNT] =
  { KAAPI_ACCESS_MODE_W, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };

  const struct kaapi_format_t* formats[PARAM_COUNT] =
  { kaapi_mem_format, kaapi_uint_format, kaapi_uint_format };

  kaapi_format_taskregister
    (format, NULL, "memset_task", sizeof(task_work_t),
     PARAM_COUNT, modes, offsets, formats, get_param_size);

  kaapi_format_taskregister_body
    (format, memset_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)memset_cuda_entry, KAAPI_PROC_TYPE_CUDA);
  
}

static void __attribute__((unused))
main_static_entry(unsigned int* base, unsigned int nelem)
{
  task_work_t* work;
  kaapi_task_t* task;
  kaapi_threadgroup_t group;

  /* register task formats */
  register_memset_task_format();
  register_add1_task_format();
  register_mul2_task_format();

#define PARTITION_COUNT 2
#define PARTITION_ID_CPU 0
#define PARTITION_ID_GPU 1

  kaapi_threadgroup_create(&group, PARTITION_COUNT);

  /* open group */
  kaapi_threadgroup_begin_partition(group);

  /* cpu partition */
  const unsigned int cpu_i = 0;
  const unsigned int cpu_j = nelem / 2;

  /* cpu::memset_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU);
  kaapi_task_initdfg(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU);

  /* cpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU);
  kaapi_task_initdfg(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU);

  /* cpu::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU);
  kaapi_task_initdfg(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU);

  /* gpu partition */
  unsigned int* const gpu_base = base + nelem / 2;
  const unsigned int gpu_i = 0;
  const unsigned int gpu_j = nelem / 2;
  const unsigned int gpu_size = (gpu_j - gpu_i) * sizeof(unsigned int);

  /* gpu::memset_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU);
  kaapi_task_initdfg(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU);

  /* gpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU);
  kaapi_task_initdfg(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU);

  /* gpu::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU);
  kaapi_task_initdfg(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU);

  /* close group */
  kaapi_threadgroup_end_partition(group);

  /* execute */
  kaapi_threadgroup_begin_execute(group);
  kaapi_threadgroup_end_execute(group);

  /* ensure memory is written back to host */
  kaapi_mem_synchronize2((kaapi_mem_addr_t)gpu_base, gpu_size);

  kaapi_threadgroup_destroy(group);
}


static void __attribute__((unused))
main_adaptive_entry(unsigned int* base, unsigned int nelem)
{
  task_work_t* work;
  kaapi_thread_t* thread;
  kaapi_task_t* task;
  kaapi_frame_t frame;

  register_add1_task_format();

  thread = kaapi_self_thread();
  kaapi_thread_save_frame(thread, &frame);

  /* processing task */
  task = kaapi_thread_toptask(thread);
  kaapi_task_init(task, add1_cpu_entry, NULL);

  work = alloc_work(thread);
  create_range(&work->range, base, 0, nelem);

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


/* check sequence */

static int check_sequence
(const unsigned int* base, unsigned int real_nelem)
{
  const unsigned int nelem = (real_nelem / THREAD_DIM_X) * THREAD_DIM_X;
  unsigned int i;

  for (i = 0; i < nelem; ++base, ++i)
  {
#define RESULT_VALUE 4
    if (*base != RESULT_VALUE)
    {
      printf("[!] check_sequence: %u, %u\n", i, *base);
      return -1;
    }
  }

  return 0;
}


/* main */

int main(int ac, char** av)
{
#define ELEM_COUNT (512 * 1000)
  static unsigned int base[ELEM_COUNT];
  unsigned int i;

  for (i = 0; i < ELEM_COUNT; ++i)
    base[i] = 0x2a;

#if CONFIG_USE_STATIC
  main_static_entry(base, ELEM_COUNT);
#else
  main_adaptive_entry(base, ELEM_COUNT);
#endif

  check_sequence(base, ELEM_COUNT);

  return 0;
}
