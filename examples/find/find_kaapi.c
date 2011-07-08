/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 ** 
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 ** 
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 ** 
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 ** 
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 ** 
 */
#include "kaapi.h"
#include <string.h>
#include <sys/types.h>
#define __USE_BSD 1
#include <sys/time.h>


/** Description of the example.
 
 Overview of the execution.
 
 What is shown in this example.
 
 Next example(s) to read.
 */
typedef struct work
{
  kaapi_workqueue_t cr;
  
  double key;
  double* array;
  kaapi_workqueue_index_t res;
  
} work_t;

/**
 */
typedef struct thief_work_t {
  double* beg;
  double* end;
  double key;
  double* res;
} thief_work_t;


/* fwd decl */
static void thief_entrypoint(void*, kaapi_thread_t*, kaapi_stealcontext_t*);


/* parallel work splitter */
static int splitter (
                     kaapi_stealcontext_t* sc, 
                     int nreq, kaapi_request_t* req, 
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
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
    kaapi_allocate_thief_result(req, sizeof(thief_work_t), NULL);
    
    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
    ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), ktr );
    tw->key = vw->key;
    tw->beg = vw->array+j-unit_size;
    tw->end = vw->array+j;
    tw->res = 0;
    
    /* initialize ktr task may be preempted before entrypoint */
    ((thief_work_t*)ktr->data)->beg = tw->beg;
    ((thief_work_t*)ktr->data)->end = tw->end;
    ((thief_work_t*)ktr->data)->res = 0;
    
    /* reply head, preempt head */
    kaapi_reply_pushhead_adaptive_task(sc, req);
  }
  
  return nrep;
}


/** thief reducer
 */
static int reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  /* victim work */
  work_t* const vw = (work_t*)varg;
  
  /* thief work */
  thief_work_t* const tw = (thief_work_t*)tdata;
  
  /* thief range continuation */
  kaapi_workqueue_index_t beg, end;
  
  /* if the master already has a result, the
   reducer purpose is only to abort thieves
   */
  if (vw->res != (kaapi_workqueue_index_t)-1)
    return 0;
  
  /* check if the thief found a result */
  if (tw->res != 0)
  {
    /* do not continue the work */
    vw->res = tw->res - vw->array;
    return 0;
  }
  
  /* otherwise, continue preempted thief work */
  beg = (kaapi_workqueue_index_t)(tw->beg - vw->array);
  end = (kaapi_workqueue_index_t)(tw->end - vw->array);
  kaapi_workqueue_set(&vw->cr, beg, end);
  
  return 0;
}


/** seq work extractor 
 */
static int extract_seq(work_t* w, double** pos, double** end)
{
  int err;

  /* extract from range beginning */
  kaapi_workqueue_index_t i, j;
  
#define CONFIG_SEQ_GRAIN 128
  if ((err =kaapi_workqueue_pop(&w->cr, &i, &j, CONFIG_SEQ_GRAIN)) !=0) return 1;
  if (i == j) return -1;
  
  *pos = w->array + i;
  *end = w->array + j;
  
  return 0; /* success */
}


/** thief entrypoint 
 */
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* input work */
  thief_work_t* const work = (thief_work_t*)args;
  
  /* resulting work */
  thief_work_t* const res_work = kaapi_adaptive_result_data(sc);
  
  /* key to find */
  const double key = work->key;
  
  /* for key in [pos, end[ */
  while (work->beg != work->end)
  {
    if (*work->beg == key)
    {
      res_work->res = work->beg;
      break ;
    }
    
    /* update prior reducing */
    ++work->beg;
    
    /* check if we have been preempted. if this is the
     case, work is copied into the ktr data and then
     passed as an argument to the reducer called by
     the master.
     note that checking for preemption should not be
     done at each step of the iteration for performance
     reasons.
     */
    const unsigned int is_preempted = kaapi_preemptpoint
    (sc, NULL, NULL, (void*)work, sizeof(thief_work_t), NULL);
    if (is_preempted)
    {
      /* we have been preempted, return. */
      return ;
    }
  }
  
  /* we are finished, update results. */
  res_work->beg = work->beg;
  res_work->end = work->end;
}


/* find main function */
static size_t find( double* array, size_t size, double key )
{
  /* return the key position, or (size_t)-1 if not found */
  
  kaapi_thread_t* thread;
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;
  work_t  work;
  double* pos, *end;
  
  /* get the self thread */
  thread = kaapi_self_thread();
  
  /* initialize work */
  kaapi_workqueue_init(&work.cr, 0, (kaapi_workqueue_index_t)size);
  work.key   = key;
  work.array = array;
  work.res   = (size_t)-1;
  
  /* push an adaptive task. set the preemption flag. */
  sc = kaapi_task_begin_adaptive(
                                 thread, 
                                 KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION, 
                                 splitter, 
                                 &work     /* arg for splitter = work to split */
                                 );
  
  /* while there is sequential work to do */
redo_work:
  while (!extract_seq(&work, &pos, &end))
  {
    /* find the key in [pos, end[ */
    for (; pos != end; ++pos)
      if (key == *pos)
      {
        /* key found, disable stealing... */
        kaapi_steal_setsplitter(sc, 0, 0);
        kaapi_workqueue_set(&work.cr,0,0);
        
        work.res = pos - array;
        /* ... and abort thieves processing */
        goto preempt_thieves;
      }
  }
  
  /* preempt and reduce thieves */
preempt_thieves:
  
  if ((ktr = kaapi_get_thief_head(sc)) != NULL)
  {
    kaapi_preempt_thief(sc, ktr, NULL, reducer, (void*)&work);
    
    /* result not found, continue the work */
    if (work.res == (size_t)-1)
      goto redo_work;
    
    /* continue until no more thief */
    goto preempt_thieves;
  }
  
  /* wait for thieves */
  kaapi_task_end_adaptive(sc);
  /* here: 1/ all thieves have finish their result */
  
  return work.res;
}


/**
 */
int main(int ac, char** av)
{
  struct timeval tms[3];
  double sum = 0.f;
  size_t i;
  
#define ITEM_COUNT 1000000
  static double array[ITEM_COUNT];
  
  /* initialize the runtime */
  kaapi_init(1, &ac, &av);
  
  for (ac = 0; ac < 1000; ++ac)
  {
    /* initialize, apply, check */
    
    for (i = 0; i < ITEM_COUNT; ++i)
      array[i] = (double)i;
    
    const double key = (double)(ITEM_COUNT - 1);
    
    gettimeofday(&tms[0], NULL);
    const size_t res = find( array, ITEM_COUNT, key );
    gettimeofday(&tms[1], NULL);
    timersub(&tms[1], &tms[0], &tms[2]);
    const double diff = (double)(tms[2].tv_sec * 1000000 + tms[2].tv_usec);
    sum += diff;
    
    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] == key)
        break ;
    if (i == ITEM_COUNT)
      i = (size_t)-1;
    
    if (i != res)
    {
      printf("invalid %lu != %lu\n", i, res);
      break ;
    }
  }
  
  printf("done: %lf\n", sum / 1000);
  
  /* finalize the runtime */
  kaapi_finalize();
  
  return 0;
}
