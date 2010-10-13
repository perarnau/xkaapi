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
#include <math.h>
#include <sys/types.h>


/** Description of the example.

    Overview of the execution.
      The previous example, for_each_0.c has a main drawback: 
    - if the work load is unbalanced, then some of the thief becomes idle and
    try to steal other threads. But an overloaded thief cannot be steal once
    it begins its execution.
    
    What is shown in this example.
      The purpose of this example is to show how to allow thief to be theft
    by other idle thread. The idea is just to declare executed task as new 
    adaptive algorithm.
    
    Next example(s) to read.
*/
typedef struct work
{
  kaapi_atomic_t lock;

  void (*op)(double*);
  double* array;

  volatile size_t beg;
  volatile size_t end;

} work_t;


/**
*/
typedef work_t thief_work_t;


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
    i = vw->end - stolen_size;
    j = vw->end;
    vw->end -= stolen_size;
  }

  unlock_work(vw);

  if (unit_size == 0)
    return 0;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* thief work */
    thief_work_t* const tw = kaapi_reply_init_adaptive_task
      ( req, (kaapi_task_body_t)thief_entrypoint, sc, 0 );
    tw->op    = vw->op;
    tw->array = vw->array+j-unit_size;
    tw->beg   = 0;
    tw->end   = unit_size;

    kaapi_reply_push_adaptive_task( req, sc );
  }

  return nrep;
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


/** thief entrypoint:
    - the extra args hodls a pointer to the implicit stealcontext where the task is running
    - at the end of the function, the entry point does not need to explicitly call 
    kaapi_task_end_adaptive, which is called into the callee.
*/
static void thief_entrypoint(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* sc)
{
  /* range to process */
  double* beg;
  double* end;

  /* process the work */
  thief_work_t* thief_work = (thief_work_t*)args;

  /* set the splitter for this task */
  kaapi_steal_setsplitter(sc, splitter, thief_work );

  /* while there is sequential work to do*/
  while (extract_seq(thief_work, &beg, &end) != -1)
  {
    /* apply w->op foreach item in [pos, end[ */
    for (; beg != end; ++beg)
      thief_work->op(beg);
  }
}




/* For each main function */
static void for_each( double* array, size_t size, void (*op)(double*) )
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
  KAAPI_ATOMIC_WRITE(&work.lock, 0);
  work.op    = op;
  work.array = array;
  work.beg   = 0;
  work.end   = size;

  /* push an adaptive task */
  sc = kaapi_task_begin_adaptive(
        thread, 
        KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION, 
        splitter, 
        &work     /* arg for splitter = work to split */
    );
  
  /* while there is sequential work to do*/
  while (extract_seq(&work, &pos, &end) != -1)
  {
    /* apply w->op foreach item in [pos, end[ */
    for (; pos != end; ++pos)
      op(pos);
  }

  /* wait for thieves */
  kaapi_task_end_adaptive(sc);
  /* here: 1/ all thieves have finish their result */
}


/**
*/
static void apply_cos( double* v )
{
  *v = cos(*v);
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
      array[i] = 0.f;

    for_each( array, ITEM_COUNT, apply_cos );

    for (i = 0; i < ITEM_COUNT; ++i)
      if (array[i] != 1.f)
      {
	printf("invalid @%lu == %lf\n", i, array[i]);
	break ;
      }
  }

  printf("done\n");

  /* finalize the runtime */
  kaapi_finalize();

  return 0;
}
