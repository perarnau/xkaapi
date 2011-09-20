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
  *i = (kaapi_workqueue_index_t)((nodeid * mi->page_size) / mi->elem_size);
  *j = *i + mi->page_pernode / mi->elem_size;
}


/**
*/
typedef work_t thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


/* parallel work splitter */

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

  const int retval = nreq;
  work_t* const vw = (work_t*)arg;
  const unsigned int size = (unsigned int)kaapi_workqueue_size(&vw->cr);

  for (; nreq; --nreq, ++req)
  {
    const unsigned int nodeid = kaapi_hws_get_request_nodeid(req);
    kaapi_workqueue_index_t i, j;

    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      (sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), 0);

    map_range(vw->mi, nodeid, &i, &j);
    if (j > size) j = size;

    printf("mapping: [%ld - %ld] @ %u\n", i, j, nodeid);

    kaapi_workqueue_init(&tw->cr, i, j);
    tw->op = vw->op;
    tw->array = vw->array;
    kaapi_reply_push_adaptive_task(sc, req);
  }

  return retval;
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
  if (range_size <= CONFIG_PAR_GRAIN)
    return 0;

  /* how much per req */
  unit_size = range_size / (nreq + 1);
  if (unit_size == 0)
  {
    nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
    unit_size = CONFIG_PAR_GRAIN;
  }

  /* perform the actual steal. if the range
     changed size in between, redo the steal
   */
  if (kaapi_workqueue_steal(&vw->cr, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* thief work */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), 0 );

    kaapi_workqueue_init(&tw->cr, j - unit_size, j);
    tw->op = vw->op;
    tw->array = vw->array;

    kaapi_reply_push_adaptive_task(sc, req);
  }

  return nrep;
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
    /* this is a hws splitter, short default splitter */
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


static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* range to process */
  double* beg;
  double* end;

  /* process the work */
  thief_work_t* thief_work = (thief_work_t*)args;

  /* set the splitter for this task */
  kaapi_steal_setsplitter(sc, splitter, thief_work );

  while (!extract_seq(thief_work, &beg, &end))
    for (; beg != end; ++beg) thief_work->op(beg);
}

/* For each main function */
static void for_each
(double* array, size_t size, void (*op)(double*), const mapping_info_t* mi)
{
  /* range to process */
  kaapi_thread_t* thread;
  kaapi_stealcontext_t* sc;
  work_t  work;
  double* pos;
  double* end;

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
  
  /* while there is sequential work to do*/
  while (!extract_seq(&work, &pos, &end))
    for (; pos != end; ++pos) op(pos);

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);
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
