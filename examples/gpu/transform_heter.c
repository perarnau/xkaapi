#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "kaapi.h"
#include "kaapi_cuda_func.h"


#define CONFIG_USE_STATIC 0
#define CONFIG_KERNEL_DIM 256


/* missing decls */
typedef struct kaapi_processor_t kaapi_processor_t;
kaapi_processor_t* kaapi_stealcontext_kproc(kaapi_stealcontext_t*);
kaapi_processor_t* kaapi_request_kproc(kaapi_request_t*);
unsigned int kaapi_request_kid(kaapi_request_t*);
unsigned int kaapi_processor_get_type(const struct kaapi_processor_t*); 
unsigned int kaapi_get_self_kid(void);


typedef struct range
{
  kaapi_access_t base;
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

typedef struct task_work
{
  volatile long lock;
  range_t range;
} task_work_t;


static task_work_t* alloc_work(kaapi_thread_t* thread)
{
  task_work_t* const work = kaapi_thread_pushdata_align
    (thread, sizeof(task_work_t), 8);

  work->range.base.data = NULL;
  work->range.i = 0;
  work->range.j = 0;

  work->lock = 0;

  return work;
}

/* task bodies */

/* add1 task */

typedef uintptr_t kaapi_mem_addr_t;
extern int kaapi_mem_synchronize2(kaapi_mem_addr_t, size_t);

static void add1_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> add1_cuda_entry [%u] 0x%lx [%u - %u[\n",
	 kaapi_get_self_kid(), (uintptr_t)base, range->i, range->j);

#if 1 /* driver api */
  {
#define THREAD_DIM_X CONFIG_KERNEL_DIM
    static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
    static const kaapi_cuda_dim2_t bdim = {1, 1};

    kaapi_cuda_func_t fn;

    kaapi_cuda_func_init(&fn);

#define STERN "transform_heter"
    kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "add1");

    /* since base points to the start of the range */
    kaapi_cuda_func_push_ptr(&fn, base);
    kaapi_cuda_func_push_uint(&fn, 0);
    kaapi_cuda_func_push_uint(&fn, range->j - range->i);

    kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

    kaapi_cuda_func_unload_ptx(&fn);
  }
#else /* c++ api */
  {
    add1_heter<<<1, CONFIG_KERNEL_DIM, 0, stream>>>
      (base, range->i, range->j);
  }
#endif /* driver api */

  printf("< add1_cuda_entry\n");
}

static void add1_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  range_t* const range = &((task_work_t*)arg)->range;
  unsigned int* const base = range->base.data;

  unsigned int i;

  printf("> add1_cpu_entry [%u] [%u - %u[\n",
	 kaapi_get_self_kid(), range->i, range->j);

  for (i = range->i; i < range->j; ++i)
    base[i] += 1;

  printf("< add1_cpu_entry\n");
}

/* mul2 task */

static void mul2_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> mul2_cuda_entry [%u] [%u - %u[\n",
	 kaapi_get_self_kid(), range->i, range->j);

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

  printf("> mul2_cpu_entry [%u] [%u - %u[ (%p)\n",
	 kaapi_get_self_kid(), range->i, range->j, (void*)base);

  unsigned int i;
  for (i = range->i; i < range->j; ++i)
    base[i] *= 2;

  printf("< mul2_cpu_entry\n");
}

/* memset task */

static void memset_cuda_entry
(CUstream stream, void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> memset_cuda_entry [%u] [%u - %u[ 0x%lx\n",
	 kaapi_get_self_kid(), range->i, range->j, (uintptr_t)base);

#define MEMSET_VALUE 1
  CUresult res = cuMemsetD32(base, MEMSET_VALUE, get_range_size(range));
  if (res != CUDA_SUCCESS)
    printf("cudaError 0x%lx %u\n", (uintptr_t)base, res);

  printf("< memset_cuda_entry\n");
}

static void memset_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  unsigned int* const base = (unsigned int*)range->base.data;
  unsigned int i;

  printf("> memset_cpu_entry [%u] [%u - %u[\n",
	 kaapi_get_self_kid(), range->i, range->j);

  for (i = 0; i <  range->j; ++i)
    base[i] = MEMSET_VALUE;

  printf("< memset_cpu_entry\n");
}


/* check task */

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

static void check_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  unsigned int* const base = (unsigned int*)range->base.data;

  printf("> check_cpu_entry [%u - %u[ (%p)\n", range->i, range->j, (void*)base);

  check_sequence(base, range->j - range->i);

  printf("< check_cpu_entry\n");
}


/* task formatting */

#define PARAM_COUNT 3

static size_t get_count_params
(const struct kaapi_format_t* f, const void* p)
{ return PARAM_COUNT; }

static void* get_off_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  static const kaapi_offset_t param_offsets[PARAM_COUNT] =
  {
    offsetof(task_work_t, range.base),
    offsetof(task_work_t, range.i),
    offsetof(task_work_t, range.j)
  };
  return (void*)((uintptr_t)p + param_offsets[i]);
}

static kaapi_access_mode_t get_write_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  /* memset task first param modes is write */
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_W, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };
  return modes[i];
}

static kaapi_access_mode_t get_read_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  /* check task first param modes is read */
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };
  return modes[i];
}

static kaapi_access_mode_t get_readwrite_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_RW, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };
  return modes[i];
}

static kaapi_access_t get_access_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  task_work_t* const w = (task_work_t*)p;

  kaapi_access_t a;
  a.data = (unsigned int*)w->range.base.data + w->range.i;
  a.version = w->range.base.version;
  return a;
}

static void set_access_param
(const struct kaapi_format_t* f, unsigned int i, void* p, const kaapi_access_t* a)
{
  task_work_t* const w = (task_work_t*)p;
  w->range.base.data = a->data;
  w->range.base.version = a->version;
}

static const struct kaapi_format_t* get_fmt_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
#define kaapi_mem_format kaapi_ulong_format
  const struct kaapi_format_t* formats[PARAM_COUNT] =
    { kaapi_mem_format, kaapi_uint_format, kaapi_uint_format };
  return formats[i];
}

static size_t get_size_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  switch (i)
  {
  case 0: /* range */
  {
    const task_work_t* const work = (task_work_t*)p;
    return (size_t)get_range_size(&work->range) * sizeof(unsigned int);
  }

  default: return sizeof(unsigned int);
  }
}

static void register_add1_task_format(void)
{
  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister_func
  (
   format, NULL, "add1_task", sizeof(task_work_t),
   get_count_params,
   get_readwrite_mode_param,
   get_off_param,
   get_access_param,
   set_access_param,
   get_fmt_param,
   get_size_param
  );

  kaapi_format_taskregister_body
    (format, add1_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)add1_cuda_entry, KAAPI_PROC_TYPE_CUDA);
}

static void register_mul2_task_format(void)
{
  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister_func
  (
   format, NULL, "mul2_task", sizeof(task_work_t),
   get_count_params,
   get_readwrite_mode_param,
   get_off_param,
   get_access_param,
   set_access_param,
   get_fmt_param,
   get_size_param
  );

  kaapi_format_taskregister_body
    (format, mul2_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)mul2_cuda_entry, KAAPI_PROC_TYPE_CUDA);
}


static void register_memset_task_format(void)
{
  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister_func
  (
   format, NULL, "memset_task", sizeof(task_work_t),
   get_count_params,
   get_write_mode_param,
   get_off_param,
   get_access_param,
   set_access_param,
   get_fmt_param,
   get_size_param
  );

  kaapi_format_taskregister_body
    (format, memset_cpu_entry, KAAPI_PROC_TYPE_CPU);

  kaapi_format_taskregister_body
    (format, (kaapi_task_body_t)memset_cuda_entry, KAAPI_PROC_TYPE_CUDA);
  
}

static void register_check_task_format(void)
{
  struct kaapi_format_t* const format = kaapi_format_allocate();

  kaapi_format_taskregister_func
  (
   format, NULL, "check_task", sizeof(task_work_t),
   get_count_params,
   get_read_mode_param,
   get_off_param,
   get_access_param,
   set_access_param,
   get_fmt_param,
   get_size_param
  );

  kaapi_format_taskregister_body
    (format, check_cpu_entry, KAAPI_PROC_TYPE_CPU);
}


#if CONFIG_USE_STATIC /* main static entry */

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
  register_check_task_format();

#define PARTITION_ID_CPU_0 0
#define PARTITION_ID_CPU_1 1
#define PARTITION_ID_GPU_0 2
#define PARTITION_ID_GPU_1 3
#define PARTITION_COUNT 4

  kaapi_threadgroup_create(&group, PARTITION_COUNT);

  /* open group */
  kaapi_threadgroup_begin_partition(group);

  /* cpu partition */
  const unsigned int cpu_i = 0;
  const unsigned int cpu_j = nelem / 2;

  /* cpu::memset_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_0));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_initdfg(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* cpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_1));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_1);
  kaapi_task_initdfg(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_1);

  /* cpu::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_0));
  create_range(&work->range, base, cpu_i, cpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_initdfg(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* gpu partition */
  unsigned int* const gpu_base = base + nelem / 2;
  const unsigned int gpu_i = 0;
  const unsigned int gpu_j = nelem / 2;
  const unsigned int gpu_size = (gpu_j - gpu_i) * sizeof(unsigned int);

  /* gpu::memset_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_0));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_0);
  kaapi_task_initdfg(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_0);

  /* gpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_0));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_0);
  kaapi_task_initdfg(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_0);

  /* gpu1::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_1));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_1);
  kaapi_task_initdfg(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_1);

  /* cpu::check_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_0));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_initdfg(task, check_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* cpu2::check_task, same as above but on other partition */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_1));
  create_range(&work->range, gpu_base, gpu_i, gpu_j);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_1);
  kaapi_task_initdfg(task, check_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_1);

  /* close group */
  kaapi_threadgroup_end_partition(group);

  /* execute */
  kaapi_threadgroup_begin_execute(group);
  kaapi_threadgroup_end_execute(group);

  /* ensure memory is written back to host */
  kaapi_mem_synchronize2((kaapi_mem_addr_t)gpu_base, gpu_size);

  kaapi_threadgroup_destroy(group);
}

#else /* CONFIG_USE_STATIC == 0 */

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
  const unsigned int range_size = get_range_size(range);
  if (range_size == 0) return -1;

  /* succeed even if size too large */
  if (range_size < size)
    size = range_size;

  sub->i = range->i;
  sub->j = range->i + size;
  range->i += size;

  return 0;
}

static int splitter
(kaapi_stealcontext_t* sc, int reqcount, kaapi_request_t* reqs, void* arg)
{
  task_work_t* const vwork = (task_work_t*)arg;
  unsigned int rangesize;
  unsigned int unitsize;
  range_t subrange;
  int stealres = -1;
  int repcount = 0;

  lock_work(vwork);

  rangesize = get_range_size(&vwork->range);
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
    task_work_t* const twork = kaapi_reply_init_adaptive_task
      (sc, reqs, (kaapi_task_body_t)add1_cpu_entry, sizeof(task_work_t), NULL);

    twork->lock = 0;
    kaapi_access_init(&twork->range.base, vwork->range.base.data);
    split_range(&twork->range, &subrange, unitsize);

    kaapi_reply_pushhead_adaptive_task(sc, reqs);

    --reqcount;
    ++repcount;
  }

  return repcount;
}

static int next_seq(task_work_t* work)
{
  unsigned int* const base = (unsigned int*)work->range.base.data;

  int stealres;
  range_t subrange;
  unsigned int i;

  lock_work(work);
  stealres = split_range(&subrange, &work->range, 512);
  unlock_work(work);

  if (stealres == -1) return -1;

  for (i = subrange.i; i < subrange.j; ++i)
    base[i] += 1;

  return 0;
}

static void __attribute__((unused))
main_adaptive_entry(unsigned int* base, unsigned int nelem)
{
  kaapi_thread_t* const thread = kaapi_self_thread();

  task_work_t* work;
  kaapi_stealcontext_t* ksc;

  /* initialize the sequence */
  size_t i;
  for (i = 0; i < nelem; ++i) base[i] = 3;

  register_add1_task_format();

  work = alloc_work(thread);
  create_range(&work->range, base, 0, nelem);

  ksc = kaapi_task_begin_adaptive
    (thread, KAAPI_SC_CONCURRENT, splitter, work);

  while (next_seq(work) != -1)
    ;

  kaapi_task_end_adaptive(ksc);
}

#endif /* CONFIG_USE_STATIC */


/* main */

int main(int ac, char** av)
{
#define ELEM_COUNT (CONFIG_KERNEL_DIM * 1000)
  static unsigned int base[ELEM_COUNT];
  unsigned int i;

  if (kaapi_init())
  {
    printf("initialization failure\n");
    return -1;
  }

  for (i = 0; i < ELEM_COUNT; ++i)
    base[i] = 0x2a;

#if CONFIG_USE_STATIC
  main_static_entry(base, ELEM_COUNT);
#else
  main_adaptive_entry(base, ELEM_COUNT);
#endif

  check_sequence(base, ELEM_COUNT);

  kaapi_finalize();

  return 0;
}
