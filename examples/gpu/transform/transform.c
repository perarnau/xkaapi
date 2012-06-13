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
typedef uintptr_t kaapi_mem_addr_t;
extern kaapi_processor_t* kaapi_stealcontext_kproc(kaapi_stealcontext_t*);
extern kaapi_processor_t* kaapi_request_kproc(kaapi_request_t*);
extern unsigned int kaapi_request_kid(kaapi_request_t*);
extern unsigned int kaapi_processor_get_type(const struct kaapi_processor_t*); 
extern unsigned int kaapi_get_self_kid(void);
extern void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);


typedef struct range
{
  kaapi_access_t base;
  size_t size;
} range_t;


static void create_range
(range_t* range, unsigned int* base, size_t size)
{
  kaapi_access_init(&range->base, (void*)base);
  range->size = size;
}


static inline unsigned int get_range_size(const range_t* range)
{ return range->size; }

typedef struct task_work
{
  volatile long lock;
  range_t range;
  enum { MEMSET, ADD1, MUL2 } func;
} task_work_t;


static task_work_t* alloc_work(kaapi_thread_t* thread)
{
  task_work_t* const work = kaapi_thread_pushdata_align
    (thread, sizeof(task_work_t), 8);

  work->range.base.data = NULL;
  work->range.size = 0;

  work->lock = 0;

  return work;
}

/* task bodies */

/* add1 task */

static void add1_cuda_entry
(void* arg, CUstream stream)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> add1_cuda_entry [%u] 0x%lx, %lu\n",
	 kaapi_get_self_kid(), (uintptr_t)base, range->size);

#if 1 /* driver api */
  {
#define THREAD_DIM_X CONFIG_KERNEL_DIM
    static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
    static const kaapi_cuda_dim2_t bdim = {1, 1};

    kaapi_cuda_func_t fn;

    kaapi_cuda_func_init(&fn);

#define STERN "transform"
    kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "add1");

    /* since base points to the start of the range */
    kaapi_cuda_func_push_ptr(&fn, base);
    kaapi_cuda_func_push_uint(&fn, get_range_size(range));

    kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

    kaapi_cuda_func_unload_ptx(&fn);
  }
#else /* c++ api */
  {
    add1<<<1, CONFIG_KERNEL_DIM, 0, stream>>>
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

  printf("> add1_cpu_entry [%u] %lu\n", kaapi_get_self_kid(), range->size);

  for (i = 0; i < range->size; ++i) base[i] += 1;

  printf("< add1_cpu_entry\n");
}

/* mul2 task */

static void mul2_cuda_entry
(void* arg, CUstream stream)
{
  task_work_t* const work = (task_work_t*)arg;
  range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> mul2_cuda_entry [%u] %lu\n", kaapi_get_self_kid(), range->size);

  static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
  static const kaapi_cuda_dim2_t bdim = {1, 1};

  kaapi_cuda_func_t fn;

  kaapi_cuda_func_init(&fn);

  kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "mul2");

  kaapi_cuda_func_push_ptr(&fn, base);
  kaapi_cuda_func_push_uint(&fn, get_range_size(range));

  kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

  kaapi_cuda_func_unload_ptx(&fn);

  printf("< mul2_cuda_entry\n");
}

static void mul2_cpu_entry(void* arg, kaapi_thread_t* thread)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  unsigned int* const base = range->base.data;

  printf("> mul2_cpu_entry [%u] %lu\n", kaapi_get_self_kid(), range->size);

  unsigned int i;
  for (i = 0; i < range->size; ++i) base[i] *= 2;

  printf("< mul2_cpu_entry\n");
}

/* memset task */

static void memset_cuda_entry
(void* arg, CUstream stream)
{
  task_work_t* const work = (task_work_t*)arg;
  const range_t* const range = &work->range;
  CUdeviceptr base = (CUdeviceptr)(uintptr_t)range->base.data;

  printf("> memset_cuda_entry [%u] %lu\n", kaapi_get_self_kid(), range->size);

#define MEMSET_VALUE 3
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

  printf("> memset_cpu_entry [%u] %lu\n", kaapi_get_self_kid(), range->size);

  unsigned int i;
  for (i = 0; i < range->size; ++i) base[i] = MEMSET_VALUE;

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
#define RESULT_VALUE 8
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

  printf("> check_cpu_entry %lu (%p)\n", range->size, (void*)base);

  check_sequence(base, range->size);

  printf("< check_cpu_entry\n");
}


/* task formatting */

/* the whole range is viewed as a single param */
#define PARAM_COUNT 1

static size_t get_count_params
(const struct kaapi_format_t* f, const void* p)
{ return PARAM_COUNT; }

static void* get_off_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  static const kaapi_offset_t param_offsets[PARAM_COUNT] =
    { offsetof(task_work_t, range) };
  return (void*)((uintptr_t)p + param_offsets[i]);
}

static kaapi_access_mode_t get_write_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  /* memset task first param modes is write */
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_W };
  return modes[i];
}

static kaapi_access_mode_t get_read_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  /* check task first param modes is read */
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_R };
  return modes[i];
}

static kaapi_access_mode_t get_readwrite_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_RW };
  return modes[i];
}

static kaapi_access_t get_access_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  task_work_t* const w = (task_work_t*)p;

  kaapi_access_t a;
  a.data = (unsigned int*)w->range.base.data;
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
    { kaapi_mem_format };
  return formats[i];
}

static size_t get_size_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  const task_work_t* const work = (task_work_t*)p;
  return (size_t)get_range_size(&work->range) * sizeof(unsigned int);
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
   get_size_param,
   0
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
   get_size_param,
   0
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
   get_size_param,
   0
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
   get_size_param,
   0
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
  create_range(&work->range, base + cpu_i, cpu_j - cpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_init(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* cpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_1));
  create_range(&work->range, base + cpu_i, cpu_j - cpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_1);
  kaapi_task_init(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_1);

  /* cpu::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_0));
  create_range(&work->range, base + cpu_i, cpu_j - cpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_init(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* gpu partition */
  unsigned int* const gpu_base = base + nelem / 2;
  const unsigned int gpu_i = 0;
  const unsigned int gpu_j = nelem / 2;

  /* gpu::memset_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_0));
  create_range(&work->range, gpu_base + gpu_i, gpu_j - gpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_0);
  kaapi_task_init(task, memset_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_0);

  /* gpu::add1_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_0));
  create_range(&work->range, gpu_base + gpu_i, gpu_j - gpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_0);
  kaapi_task_init(task, add1_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_0);

  /* gpu1::mul2_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_GPU_1));
  create_range(&work->range, gpu_base + gpu_i, gpu_j - gpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_GPU_1);
  kaapi_task_init(task, mul2_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_GPU_1);

  /* cpu::check_task */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_0));
  create_range(&work->range, gpu_base + gpu_i, gpu_j - gpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_0);
  kaapi_task_init(task, check_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_0);

  /* cpu2::check_task, same as above but on other partition */
  work = alloc_work(kaapi_threadgroup_thread(group, PARTITION_ID_CPU_1));
  create_range(&work->range, gpu_base + gpu_i, gpu_j - gpu_i);
  task = kaapi_threadgroup_toptask(group, PARTITION_ID_CPU_1);
  kaapi_task_init(task, check_cpu_entry, (void*)work);
  kaapi_threadgroup_pushtask(group, PARTITION_ID_CPU_1);

  /* close group */
  kaapi_threadgroup_end_partition(group);

  /* execute */
  kaapi_threadgroup_begin_execute(group);
  kaapi_threadgroup_end_execute(group);

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

  sub->base.data = (unsigned int*)range->base.data + range->size - size;
  sub->size = size;

  range->size -= size;

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

  sub->base.data = (unsigned int*)range->base.data + range->size - size;
  sub->size = size;

  range->size -= size;

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
    /* algo func to bodies */
    const kaapi_task_body_t to_bodies[] =
    {
      (kaapi_task_body_t)memset_cpu_entry,
      (kaapi_task_body_t)add1_cpu_entry,
      (kaapi_task_body_t)mul2_cpu_entry
    };

    task_work_t* const twork = kaapi_reply_init_adaptive_task
      (sc, reqs, to_bodies[vwork->func], sizeof(task_work_t), NULL);

    twork->func = vwork->func;
    twork->lock = 0;
    kaapi_access_init(&twork->range.base, 0);
    split_range(&twork->range, &subrange, unitsize);

    kaapi_reply_pushhead_adaptive_task(sc, reqs);

    --reqcount;
    ++repcount;
  }

  return repcount;
}

static int next_seq(task_work_t* work)
{
  int stealres;
  range_t subrange;
  unsigned int i;

  lock_work(work);
  stealres = split_range(&subrange, &work->range, 512);
  unlock_work(work);

  if (stealres == -1) return -1;

  unsigned int* const base = (unsigned int*)subrange.base.data;
  for (i = 0; i < subrange.size; ++i)
  {
    if (work->func == MEMSET)
      base[i] = MEMSET_VALUE;
    else if (work->func == ADD1)
      base[i] += 1;
    else if (work->func == MUL2)
      base[i] *= 2;
  }

  return 0;
}

static void __attribute__((unused))
main_adaptive_entry(unsigned int* base, unsigned int nelem)
{
  kaapi_thread_t* const thread = kaapi_self_thread();

  task_work_t* work;
  kaapi_stealcontext_t* ksc;

  work = alloc_work(thread);

  /* memset adaptive task */
  {
    /* register format */
    register_memset_task_format();

    /* init range */
    create_range(&work->range, base, nelem);
    work->func = MEMSET;

    /* start adaptive algorithm */
    ksc = kaapi_task_begin_adaptive
      (thread, KAAPI_SC_CONCURRENT, splitter, work);

    /* sequential work */
    while (next_seq(work) != -1)
      ;

    kaapi_task_end_adaptive( thread, ksc);

    /* wait for thieves */
    kaapi_sched_sync();
  }

  /* add1 adaptive task */
  {
    register_add1_task_format();

    create_range(&work->range, base, nelem);
    work->func = ADD1;

    ksc = kaapi_task_begin_adaptive
      (thread, KAAPI_SC_CONCURRENT, splitter, work);

    while (next_seq(work) != -1)
      ;

    kaapi_task_end_adaptive(thread, ksc);

    /* wait for thieves */
    kaapi_sched_sync();
  }

  /* mul2 adaptive task */
  {
    register_mul2_task_format();

    create_range(&work->range, base, nelem);
    work->func = MUL2;

    ksc = kaapi_task_begin_adaptive
      (thread, KAAPI_SC_CONCURRENT, splitter, work);

    while (next_seq(work) != -1)
      ;

    kaapi_task_end_adaptive(thread, ksc);

    /* wait for thieves */
    kaapi_sched_sync();
  }
}

#endif /* CONFIG_USE_STATIC */


/* main */

int main(int ac, char** av)
{
#define ELEM_COUNT (CONFIG_KERNEL_DIM * 10000)
  static unsigned int base[ELEM_COUNT];

  kaapi_init();

#if CONFIG_USE_STATIC
  main_static_entry(base, ELEM_COUNT);
#else
  main_adaptive_entry(base, ELEM_COUNT);
#endif

  /* use this to delete kaapi mappings. For instance,
     if the user free() then malloc(), associated kaapi
     mapping is no longer valid.
     It releases the remotely allocated memory too.
  */
  kaapi_mem_delete_host_mappings((kaapi_mem_addr_t)base, sizeof(base));

  check_sequence(base, ELEM_COUNT);

  kaapi_finalize();

  return 0;
}
