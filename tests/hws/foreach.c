#include "kaapi.h"
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/types.h>


struct mapping_info;

typedef struct work
{
  kaapi_workqueue_t cr;
  void (*op)(double*);
  double* array;
  const struct mapping_info* mi;
} work_t;


/* array allocation routines */

typedef struct mapping_info
{
  unsigned int page_size;
  unsigned int page_pernode;
  unsigned int elem_size;
} mapping_info_t;


static double* allocate_double_array
(
 unsigned int item_count,
 mapping_info_t* mi
)
{
  static const size_t page_size = 0x1000;
  const size_t size = item_count * sizeof(double);
  const size_t node_count = kaapi_hws_get_node_count(KAAPI_HWS_LEVELID_NUMA);
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
      bind_size = size - (node_count - 1) * mi->page_pernode;
    kaapi_numa_bind(addr, bind_size, i);
  }

  return (double*)p;
}

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

#if 0
    printf("push %lx - %lx @ %u\n", i, j, nodeid);
#endif

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

 redo_steal:
  /* do not steal if range size <= PAR_GRAIN */
#define CONFIG_PAR_GRAIN 128
  range_size = kaapi_workqueue_size(&vw->cr);
  if (range_size == 0) return KAAPI_WS_ERROR_EMPTY;

  /* how much per req */
  unit_size = range_size / nreq;
  if (unit_size == 0)
  {
    nreq = range_size / CONFIG_PAR_GRAIN;
    unit_size = CONFIG_PAR_GRAIN;
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
   */
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, nreq * unit_size))
    goto redo_steal;

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

#if 0
  {
    printf("[%u] %s: 0x%lx, 0x%lx\n",
	   kaapi_get_self_kid(),
	   __FUNCTION__,
	   thief_work->cr.beg,
	   thief_work->cr.end);
  }
#endif

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

  /* initialize the runtime */
  kaapi_init(1, &ac, &av);

#define ITEM_COUNT 100000
  array = allocate_double_array(ITEM_COUNT, &mi);

#if 0
  printf("RANGE: %lx - %lx\n", 0, ITEM_COUNT);
#endif
  
  for (iter = 0; iter < 100; ++iter)
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

  /* finalize the runtime */
  kaapi_finalize();
  
  return 0;
}
