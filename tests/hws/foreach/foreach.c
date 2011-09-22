/* gcc -DCONFIG_KAAPI=1 -O3 -Wall a.out.kaapi foreach.c -lkaapi -lm */
#ifndef CONFIG_KAAPI
# define CONFIG_KAAPI 0
#endif

/* gcc -DCONFIG_OMP=1 -fopenmp -O3 -Wall foreach.c -lnuma -lm */
#ifndef CONFIG_OMP
# define CONFIG_OMP 0
#endif

#ifndef CONFIG_IDKOIFF
# define CONFIG_IDKOIFF 0
#endif

#ifndef CONFIG_IDFREEZE
# define CONFIG_IDFREEZE 0
#endif

#if CONFIG_IDFREEZE
# define CONFIG_MAX_PROC 48
# define CONFIG_MAX_NODE 8 /* numa node count */
# define CONFIG_CACHE_SIZE (5 * 1024 * 1024)
#elif CONFIG_IDKOIFF
# define CONFIG_MAX_PROC 16
# define CONFIG_MAX_NODE 8
# define CONFIG_CACHE_SIZE (1024 * 1024)
#else
# error "undefined host"
#endif

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/types.h>

#if CONFIG_KAAPI
#include "kaapi.h"
#endif


#if CONFIG_OMP

#include <numaif.h>
#include <numa.h>
#include <errno.h>
#include <stdint.h>
#include <sys/time.h>

static int kaapi_numa_bind(const void* addr, size_t size, int node)
{
  static const unsigned long maxproc = 64;

  const int mode = MPOL_BIND;
  const unsigned int flags = MPOL_MF_STRICT | MPOL_MF_MOVE;
  const unsigned long maxnode = CONFIG_MAX_PROC;
  
  if ((size & 0xfff) !=0) /* should be divisible by 4096 */
  {
    errno = EINVAL;
    return -1;
  }
  if (((uintptr_t)addr & 0xfff) !=0) /* should aligned on page boundary */
  {
    errno = EFAULT;
    return -1;
  }
  
  unsigned long nodemask
    [(maxproc + (8 * sizeof(unsigned long) -1))/ (8 * sizeof(unsigned long))];

  memset(nodemask, 0, sizeof(nodemask));

  nodemask[node / (8 * sizeof(unsigned long))] |=
    1UL << (node % (8 * sizeof(unsigned long)));

  if (mbind((void*)addr, size, mode, nodemask, maxnode, flags))
  {
    perror("mbind");
    return -1;
  }

  return 0;
}

static uint64_t kaapi_get_elapsedns(void)
{
  struct timeval tv;
  uint64_t retval = 0;
  int err = gettimeofday( &tv, 0);
  if (err  !=0) return 0;
  retval = (uint64_t)tv.tv_sec;
  retval *= 1000000UL;
  retval += (uint64_t)tv.tv_usec;
  return retval*1000UL;
}

#endif /* CONFIG_OMP */


/* array allocation routines */

typedef struct mapping_info
{
  unsigned int page_size;
  unsigned int page_pernode;
  unsigned int elem_size;
} mapping_info_t;


static double* allocate_double_array
(size_t item_count, mapping_info_t* mi)
{
  static const size_t page_size = 0x1000;
  const size_t size = item_count * sizeof(double);
#if CONFIG_OMP
  const size_t node_count = CONFIG_MAX_NODE;
#else
  const size_t node_count = kaapi_hws_get_node_count(KAAPI_HWS_LEVELID_NUMA);
#endif
  size_t i;
  double* p;

  /* allocate */
  if (posix_memalign((void**)&p, page_size, size))
  {
    perror("posix_memalign");
    exit(-1);
  }

  /* init mapping info */
  mi->page_size = page_size;
  mi->page_pernode = size / (mi->page_size * node_count);
  if (mi->page_size * node_count * mi->page_pernode != size)
    mi->page_pernode += 1;
  mi->elem_size = sizeof(double);

  /* bind the pages */
  for (i = 0; i < node_count; ++i)
  {
    const void* const addr = (const void*)
      ((uintptr_t)p + i * mi->page_pernode * mi->page_size);
    size_t bind_size = mi->page_pernode * mi->page_size;
    if (i == (node_count - 1))
    {
      bind_size = size - bind_size * (node_count - 1);
      if (bind_size & (mi->page_size - 1UL))
	bind_size = (bind_size + mi->page_size) & ~(mi->page_size - 1UL);
    }

#if 1
    if (kaapi_numa_bind(addr, bind_size, i))
    {
      printf("[!] kaapi_numa_bind\n");
      exit(-1);
    }
#endif

#if 0
    /* check the page mapping */
    {
      uintptr_t fu = (uintptr_t)addr;
      uintptr_t bar = fu + bind_size;
      for (; fu < bar; fu += mi->page_size)
      {
	const int nodeid = kaapi_numa_get_page_node(fu);
	if (nodeid != i)
	{
	  printf("invalid mapping %lx@%u!=@%u\n", fu, nodeid, i);
	  exit(-1);
	}
      }
    }
#endif
  }

  return (double*)p;
}


static void init_double_array(double* p, unsigned int n)
{
  memset(p, 0, n * sizeof(double));
}


#if CONFIG_KAAPI

typedef struct work
{
  kaapi_workqueue_t cr;
  void (*op)(double*);
  double* array;
  const struct mapping_info* mi;
} work_t;


static void map_range
(
 const mapping_info_t* mi,
 unsigned int nodeid,
 kaapi_workqueue_index_t* i,
 kaapi_workqueue_index_t* j
)
{
  /* get i and j such that array[i, j] is allocated on nodeid */

  const unsigned long elem_per_node =
    (mi->page_pernode * mi->page_size) / mi->elem_size;

  *i = (kaapi_workqueue_index_t)(nodeid * elem_per_node);
  *j = *i + elem_per_node;
}

typedef work_t thief_work_t;


static void thief_entrypoint(void*, kaapi_thread_t*);

static int splitter(kaapi_stealcontext_t*, int, kaapi_request_t*, void*);


static int do_hws_splitter
(
 kaapi_stealcontext_t* sc,
 int nreq,
 kaapi_request_t* req,
 void* arg,
 kaapi_hws_levelid_t levelid
)
{
  /* some notes on hws splitting:
     . there is one request per levelid node
     . the algorithm is not yet published, so non concurrency issues
  */

  work_t* const vw = (work_t*)arg;
  const unsigned int size = (unsigned int)kaapi_workqueue_size(&vw->cr);

#define KAAPI_WS_ERROR_SUCCESS 0
#define KAAPI_WS_ERROR_EMPTY 1
  if (size == 0) return KAAPI_WS_ERROR_EMPTY;

  for (; nreq; --nreq, ++req)
  {
    const unsigned int nodeid = kaapi_hws_get_request_nodeid(req);
    kaapi_workqueue_index_t i, j;

    thief_work_t* const tw = kaapi_hws_init_adaptive_task
      (sc, req, thief_entrypoint, sizeof(thief_work_t), splitter);

    map_range(vw->mi, nodeid, &i, &j);
    if (j > size) j = size;

    kaapi_workqueue_init(&tw->cr, i, j);
    tw->op = vw->op;
    tw->array = vw->array;

    /* commit the reply */
    kaapi_hws_reply_adaptive_task(sc, req);
  }

  return KAAPI_WS_ERROR_SUCCESS;
}


static int do_flat_splitter
(
 kaapi_stealcontext_t* sc,
 int nreq,
 kaapi_request_t* req,
 void* args
)
{
  /* victim work */
  work_t* const vw = (work_t*)args;

  /* stolen range */
  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;

  /* reply count */
  int nrep = 0;

  /* size per request */
  kaapi_workqueue_index_t unit_size;

  /* do not steal if range size <= PAR_GRAIN */
  range_size = kaapi_workqueue_size(&vw->cr);
  if (range_size == 0) return KAAPI_WS_ERROR_EMPTY;

#define CONFIG_PAR_GRAIN 256
  if (range_size < CONFIG_PAR_GRAIN)
  {
    unit_size = range_size;
    nreq = 1;
  }
  else if (nreq == 1)
  {
    unit_size = range_size / 2;
  }
  else /* how much per req */
  {
    /* equally sized among nreq */
    unit_size = range_size / nreq;
    if (unit_size == 0)
    {
      nreq = range_size / CONFIG_PAR_GRAIN;
      unit_size = CONFIG_PAR_GRAIN;
    }
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
   */
  kaapi_workqueue_steal(&vw->cr, &i, &j, nreq * unit_size);

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    thief_work_t* const tw = kaapi_hws_init_adaptive_task
      (sc, req, thief_entrypoint, sizeof(thief_work_t), splitter);

    kaapi_workqueue_init(&tw->cr, j - unit_size, j);
    tw->op = vw->op;
    tw->array = vw->array;

    /* commit the reply */
    kaapi_hws_reply_adaptive_task(sc, req);
  }

  return KAAPI_WS_ERROR_SUCCESS;
}


static int splitter
(
 kaapi_stealcontext_t* sc, 
 int nreq,
 kaapi_request_t* req, 
 void* args
)
{
  /* test for hws splitter */
  kaapi_hws_levelid_t levelid;
  if (kaapi_hws_get_splitter_info(sc, &levelid) == 0)
  {
    /* this is a hws splitter, shunt the default splitter */
    kaapi_hws_clear_splitter_info(sc);
    return do_hws_splitter(sc, nreq, req, args, levelid);
  }

  /* flat splitter */
  return do_flat_splitter(sc, nreq, req, args);
}


static int extract_seq(work_t* w, double** pos, double** end)
{
  /* extract from range beginning */
  kaapi_workqueue_index_t i, j;
  
#define CONFIG_SEQ_GRAIN 256
  if (kaapi_workqueue_pop(&w->cr, &i, &j, CONFIG_SEQ_GRAIN)) return 1;
  
  *pos = w->array + i;
  *end = w->array + j;
  
  return 0; /* success */
}


static void thief_entrypoint(void* args, kaapi_thread_t* thread)
{
  /* range to process */
  double* beg;
  double* end;

  /* process the work */
  thief_work_t* thief_work = (thief_work_t*)args;

  while (!extract_seq(thief_work, &beg, &end))
    for (; beg != end; ++beg) thief_work->op(beg);
}

/* For each main function */
static void for_each
(double* array, size_t size, void (*op)(double*), const mapping_info_t* mi)
{
  /* range to process */
  kaapi_stealcontext_t* sc;
  kaapi_thread_t* thread;
  work_t work;

  /* get the self thread */
  thread = kaapi_self_thread();

  /* initialize work */
  kaapi_workqueue_init(&work.cr, 0, (kaapi_workqueue_index_t)size);
  work.op = op;
  work.array = array;
  work.mi = mi;

  /* push an adaptive task */
  sc = kaapi_task_begin_adaptive
  (
   thread, 
   KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION | KAAPI_SC_HWS_SPLITTER,
   splitter, 
   &work
  );

  kaapi_hws_end_adaptive(sc);
}

#endif /* CONFIG_XKAAPI */

#if CONFIG_OMP

/* For each main function */
static void for_each
(double* array, size_t size, void (*op)(double*), const mapping_info_t* mi)
{
  unsigned int i;

#pragma omp parallel for
  for (i = 0; i < size; ++i)
    op(array + i);
}

#endif /* CONFIG_OMP */


static void apply_cos(double* v)
{
  *v += cos(*v);
}

 
int main(int ac, char** av)
{
  double t0,t1;
  double sum = 0.f;
  size_t i;
  size_t iter;
  double* array;
  mapping_info_t mi;

#if CONFIG_KAAPI
  /* initialize the runtime */
  kaapi_init(1, &ac, &av);
#endif /* CONFIG_KAAPI */

#define TOTAL_SIZE (1 * CONFIG_MAX_PROC * 2 * CONFIG_CACHE_SIZE)
#define ITEM_COUNT (TOTAL_SIZE / sizeof(double))
  array = allocate_double_array(ITEM_COUNT, &mi);

  init_double_array(array, ITEM_COUNT);

#if 0
  printf("RANGE: %lx - %lx\n", 0, ITEM_COUNT);
#endif
  
  for (iter = 0; iter < 1; ++iter)
  {
    for (i = 0; i < ITEM_COUNT; ++i) array[i] = 0.f;

    t0 = kaapi_get_elapsedns();
    for_each(array, ITEM_COUNT, apply_cos, &mi);
    t1 = kaapi_get_elapsedns();

    sum += (t1 - t0) / 1000;

#if 0
    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] != 1.f)
      {
        printf("invalid @%lu == %lf\n", i, array[i]);
        break ;
      }
#endif
  }

  printf("done: %lf (ms)\n", sum / 100);

  free(array);

#if CONFIG_KAAPI
  /* finalize the runtime */
  kaapi_finalize();
#endif /* CONFIG_KAAPI */
  
  return 0;
}
