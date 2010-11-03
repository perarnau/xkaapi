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


/** Description of the example.

    Overview of the execution.
    
    What is shown in this example.
    
    Next example(s) to read.
*/
typedef struct work
{
  kaapi_atomic_t lock;

  double key;

  double* array;

  volatile size_t beg;
  volatile size_t end;

  size_t res;

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


/* spinlocking */
static void lock_work(work_t* w)
{
  while ( (KAAPI_ATOMIC_READ(&w->lock) == 1) || !KAAPI_ATOMIC_CAS(&w->lock, 0, 1))
    ;
}

/* unlock */
static void unlock_work(work_t* w)
{
  KAAPI_ATOMIC_WRITE(&w->lock, 0);
}


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
  size_t i, j;

  /* reply count */
  int nrep = 0;

  /* size per request */
  unsigned int unit_size;

  /* concurrent with victim */
  lock_work(vw);

  const size_t total_size = vw->end - vw->beg;

  /* how much per req */
#define CONFIG_PAR_GRAIN 128
  unit_size = 0;
  if (total_size > CONFIG_PAR_GRAIN)
  {
    unit_size = total_size / (nreq + 1);
    if (unit_size == 0)
    {
      nreq = (total_size / CONFIG_PAR_GRAIN) - 1;
      unit_size = CONFIG_PAR_GRAIN;
    }

    /* steal and update victim range */
    const size_t stolen_size = unit_size * nreq;
    i = vw->beg - stolen_size;
    j = vw->end;
    vw->end -= stolen_size;
  }

  unlock_work(vw);

  if (unit_size == 0)
    return 0;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* for reduction, a result is needed. take care of initializing it */
    kaapi_taskadaptive_result_t* const ktr =
      kaapi_allocate_thief_result(req, sizeof(thief_work_t), NULL);
    ((thief_work_t*)ktr->data)->beg = 0;
    ((thief_work_t*)ktr->data)->end = 0;
    ((thief_work_t*)ktr->data)->res = 0;

    /* thief work: not adaptive result because no preemption is used here  */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, (kaapi_task_body_t)thief_entrypoint, sizeof(thief_work_t), ktr );
    tw->key = vw->key;
    tw->beg = vw->array+j-unit_size;
    tw->end = vw->array+j;
    tw->res = 0;

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

  /* if the master already has a result, the
     reducer purpose is only to abort thieves
   */
  if (vw->res != (size_t)-1)
    return 0;

  /* check if the thief found a result */
  if (tw->res != 0)
  {
    vw->res = tw->res - vw->array;
    return 0;
  }

  /* otherwise, continue preempted thief work */
  lock_work(vw);
  vw->beg = tw->beg - vw->array;
  vw->end = tw->end - vw->array;
  unlock_work(vw);

  return 0;
}


/** seq work extractor 
*/
static int extract_seq(work_t* w, double** pos, double** end)
{
  /* extract from range beginning */
#define CONFIG_SEQ_GRAIN 64
  size_t seq_size = CONFIG_SEQ_GRAIN;

  size_t i, j;

  lock_work(w);

  i = w->beg;

  if (seq_size > (w->end - w->beg))
    seq_size = w->end - w->beg;

  j = w->beg + seq_size;
  w->beg += seq_size;

  unlock_work(w);

  if (seq_size == 0)
    return -1;

  *pos = w->array + i;
  *end = w->array + j;

  return 0;
}


/** thief entrypoint 
*/
static void thief_entrypoint
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* process the work */
  thief_work_t* const work = (thief_work_t*)args;

  /* key to find */
  const double key = work->key;

  /* for key in [pos, end[ */
  for (; work->beg < work->end; ++work->beg)
  {
    if (*work->beg == key)
    {
      work->res = work->beg;
      break ;
    }

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
  thief_work_t* const res_work = kaapi_adaptive_result_data(sc);
  res_work->beg = 0;
  res_work->end = 0;
  res_work->res = work->res;
}


/* find main function */
static size_t find( double* array, size_t size, double key )
{
  /* return the key position, or (size_t)-1 if not found */

  kaapi_thread_t* thread;
  kaapi_taskadaptive_result_t* ktr;
  kaapi_stealcontext_t* sc;
  work_t  work;
  double* pos;
  double* end;

  /* get the self thread */
  thread = kaapi_self_thread();

  /* initialize work */
  KAAPI_ATOMIC_WRITE(&work.lock, 0);
  work.key   = key;
  work.array = array;
  work.beg   = 0;
  work.end   = size;
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
  while (extract_seq(&work, &pos, &end) != -1)
  {
    /* find the key in [pos, end[ */
    for (; pos != end; ++pos)
      if (key == *pos)
      {
	/* key found, disable stealing... */
	kaapi_steal_setsplitter(sc, 0, 0);
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
  size_t i;

#define ITEM_COUNT 100000
  static double array[ITEM_COUNT];

  /* initialize the runtime */
  kaapi_init();

  for (ac = 0; ac < 1000; ++ac)
  {
    /* initialize, apply, check */

    for (i = 0; i < ITEM_COUNT; ++i)
      array[i] = (double)i;

    const double key = (double)(ITEM_COUNT - 1);

    const size_t res = find( array, ITEM_COUNT, key );

    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] == key)
	break ;
    if (i == ITEM_COUNT)
      i = (size_t)-1;
    if (i != res)
      printf("invalid %lu != %lu\n", i, res);
  }

  printf("done\n");

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}
