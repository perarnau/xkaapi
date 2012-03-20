/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** fabien.lementec@gmail.com 
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
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaapi_impl.h"
#include "kaapi_event_recorder.h"

/**/
#define KAAPI_GET_THREAD_STATE(kproc)\
  ((kproc)->curr_perf_regs == (kproc)->perf_regs[KAAPI_PERF_USER_STATE] ? KAAPI_PERF_USER_STATE : KAAPI_PERF_SCHEDULE_STATE)


/* global mask of events to register */
uint64_t kaapi_event_mask;

/** Fifo List of buffers to record
    - push in tail
    - pop in head
*/
static kaapi_event_buffer_t*  listevt_head =0;
static kaapi_event_buffer_t*  listevt_tail =0;
static pthread_mutex_t mutex_listevt;

static pthread_cond_t signal_thread;

/** List of free buffers
*/
static kaapi_event_buffer_t*  listevtfree_head =0;
static pthread_mutex_t mutex_listevtfree_head;


/** List of fd, one for each core:
    avoid to reorder buffer... if kaapi_event_reader is
    recoded, we can write all event buffers in one file.
*/
static int listfd_set[KAAPI_MAX_PROCESSOR];

#if defined(KAAPI_USE_PERFCOUNTER)
/** The thread to join in termination
*/
static pthread_t collector_threadid;
#endif

/* write one bloc. Should not be concurrent */
static void _kaapi_write_evb( kaapi_event_buffer_t* evb )
{
  int kid = evb->kid;
  if (listfd_set[kid] == -1)
  {
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/events.%s.%i.evt", getenv("USER"), kid );
    else
      sprintf(filename,"/tmp/events.%i.evt", kid );

    /* open it */
    listfd_set[kid] = open(filename, O_WRONLY|O_CREAT|O_TRUNC);
    kaapi_assert( listfd_set[kid] != -1 );
    fchmod( listfd_set[kid], S_IRUSR|S_IWUSR);
  }
  ssize_t sz_write = write(listfd_set[kid], evb->buffer, sizeof(kaapi_event_t)*evb->pos);
  kaapi_assert( sz_write == (ssize_t)(sizeof(kaapi_event_t)*evb->pos) );
  evb->pos = 0;
}

/* infinite loop to write generated buffer */
static void* _kaapi_event_flushimator(void* arg)
{
  kaapi_event_buffer_t* evb;
  while (1)
  {    
    pthread_mutex_lock(&mutex_listevt);
    while (listevt_head ==0)
    {
      if (kaapi_isterminated()) 
        goto exit_fromterm;
      pthread_cond_wait(&signal_thread, &mutex_listevt);
    }
    /* pick up atomically */
    evb = listevt_head;
    listevt_head = evb->next;
    if (listevt_head ==0)
      listevt_tail = 0;
    pthread_mutex_unlock(&mutex_listevt);
    
    evb->next = 0;
    _kaapi_write_evb(evb);
    
    /* free buffer */
    pthread_mutex_lock(&mutex_listevtfree_head);
    evb->next = listevtfree_head;
    listevtfree_head = evb;
    pthread_mutex_unlock(&mutex_listevtfree_head);    
  }

exit_fromterm:
  pthread_mutex_unlock(&mutex_listevt);
  return 0;
}


/**
*/
void kaapi_event_flushbuffer( kaapi_processor_t* kproc )
{
  kaapi_event_buffer_t* evb = kproc->eventbuffer;
  if (evb ==0) return;

  /* push buffer in listevt buffer list */
  kproc->eventbuffer = 0;
  pthread_mutex_lock(&mutex_listevt);
  evb->next = 0;
  if (listevt_head !=0)
    listevt_tail->next = evb;
  else listevt_head = evb;
  listevt_tail = evb;
  pthread_mutex_unlock(&mutex_listevt);
  pthread_cond_signal(&signal_thread);

  /* alloc new buffer */
  if (listevtfree_head ==0)
    evb = (kaapi_event_buffer_t*)malloc(sizeof(kaapi_event_buffer_t));
  else 
  {
    pthread_mutex_lock(&mutex_listevtfree_head);
    evb = listevtfree_head;
    listevtfree_head = evb->next;
    pthread_mutex_unlock(&mutex_listevtfree_head);    
  }

  evb->next = 0;
  evb->pos  = 0;
  evb->kid  = kproc->kid;

  kproc->eventbuffer = evb;
}


/*
*/
void kaapi_event_closebuffer( kaapi_processor_t* kproc )
{
  kaapi_event_buffer_t* evb = kproc->eventbuffer;
  if (evb ==0) return;

  kproc->eventbuffer = 0;
  pthread_mutex_lock(&mutex_listevt);
  evb->next = 0;
  if (listevt_head !=0)
    listevt_tail->next = evb;
  else listevt_head = evb;
  listevt_tail = evb;
  pthread_mutex_unlock(&mutex_listevt);
  pthread_cond_signal(&signal_thread);
}



/**
*/
void kaapi_perf_global_init(void)
{
  int i;
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    listfd_set[i] = -1;

#if defined(KAAPI_USE_PERFCOUNTER)
  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    pthread_mutex_init(&mutex_listevt, 0);
    pthread_mutex_init(&mutex_listevtfree_head, 0);
    pthread_cond_init(&signal_thread, 0);
    pthread_create(&collector_threadid, 0, _kaapi_event_flushimator, 0);
  }
#endif
  

  kaapi_mt_perf_init();
}


/** Finish trace. Assume that threads have reach the barrier and flush
    their event buffers.
*/
void kaapi_perf_global_fini(void)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  void* result;
  int i;
  kaapi_event_buffer_t* evb;
#endif

  /* close and flush */
  kaapi_mt_perf_fini();

  kaapi_assert( kaapi_isterminated() );

#if defined(KAAPI_USE_PERFCOUNTER)
  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    pthread_mutex_lock(&mutex_listevt);
    pthread_cond_signal(&signal_thread);
    pthread_mutex_unlock(&mutex_listevt);
    pthread_join(collector_threadid, &result);
  
    /* flush remains buffer */
    pthread_mutex_lock(&mutex_listevt);
    while (listevt_head !=0)
    {
      evb = listevt_head;
      listevt_head = evb->next;
      if (listevt_head ==0)
        listevt_tail = 0;
      evb->next = 0;
      _kaapi_write_evb(evb);
      free(evb);
    }
    pthread_mutex_unlock(&mutex_listevt);
  }
    
  /* close all file descriptors */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    if (listfd_set[i] != -1)
      close(listfd_set[i]);

  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    /* destroy mutexes/conditions */
    pthread_cond_destroy(&signal_thread);
    pthread_mutex_destroy(&mutex_listevt);
    pthread_mutex_destroy(&mutex_listevtfree_head);
  }
#endif
}


/**
*/
void kaapi_perf_thread_init(kaapi_processor_t* kproc, int isuser)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );

  if (getenv("KAAPI_RECORD_TRACE") !=0)
  {
    kproc->eventbuffer = 
      (kaapi_event_buffer_t*)malloc(sizeof(kaapi_event_buffer_t));
    kproc->eventbuffer->pos  = 0;
    kproc->eventbuffer->kid  = kproc->kid;
    kproc->eventbuffer->next = 0;
  }
  
  memset( kproc->perf_regs, 0, sizeof( kproc->perf_regs) );
  kproc->start_t[0] = kproc->start_t[1] = 0;
  kproc->curr_perf_regs = kproc->perf_regs[isuser]; 
  
  kaapi_mt_perf_thread_init( kproc, isuser );
}


/*
*/
void kaapi_perf_thread_fini(kaapi_processor_t* kproc)
{
  kaapi_mt_perf_thread_fini( kproc );
  
  if (kproc->eventbuffer !=0)
    kaapi_event_closebuffer(kproc);
}


/*
*/
void kaapi_perf_thread_start(kaapi_processor_t* kproc)
{
  kproc->start_t[KAAPI_GET_THREAD_STATE(kproc)] = kaapi_get_elapsedns();
  kaapi_mt_perf_thread_start(kproc);
}


/*
*/
void kaapi_perf_thread_stop(kaapi_processor_t* kproc)
{
  int mode;
  kaapi_perf_counter_t delay;

  kaapi_mt_perf_thread_stop(kproc);

  mode = KAAPI_GET_THREAD_STATE(kproc);
  kaapi_assert_debug( kproc->start_t[mode] != 0 ); /* else none started */
  delay = kaapi_get_elapsedns() - kproc->start_t[mode];
  KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_T1) += delay;
  kproc->start_t[mode] =0;
}


/*
*/
void kaapi_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser )
{
  kaapi_perf_counter_t date;
  
  kaapi_assert_debug( (isuser ==KAAPI_PERF_SCHEDULE_STATE)
                   || (isuser ==KAAPI_PERF_USER_STATE) );
  if (kproc->curr_perf_regs != kproc->perf_regs[isuser])
  {
    /* doit only iff real swap */
    kaapi_assert_debug( kproc->start_t[1-isuser] != 0 ); /* else none started */
    kaapi_assert_debug( kproc->start_t[isuser] == 0 );   /* else already started */
    date = kaapi_get_elapsedns();
    KAAPI_PERF_REG(kproc, KAAPI_PERF_ID_T1) += date - kproc->start_t[1-isuser];
    kproc->start_t[isuser]   = date;
    kproc->start_t[1-isuser] = 0;
    
    kaapi_mt_perf_thread_stopswapstart(kproc, isuser);

    kproc->curr_perf_regs = kproc->perf_regs[isuser]; 
  }
}



/**
*/
void _kaapi_signal_dump_counters(int xxdummy)
{
  kaapi_event_buffer_t* evb;
  uint32_t i;
  
  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    kaapi_event_closebuffer( kaapi_all_kprocessors[i] );
  }

  pthread_mutex_lock(&mutex_listevt);
  while (listevt_head !=0)
  {
    evb = listevt_head;
    listevt_head = evb->next;
    if (listevt_head ==0)
      listevt_tail = 0;
    evb->next = 0;
    _kaapi_write_evb(evb);
    free(evb);
  }
  pthread_mutex_unlock(&mutex_listevt);

  _exit(0);
}


/* convert the idset passed as an argument
*/
static const kaapi_perf_idset_t* get_perf_idset
(
  const kaapi_perf_idset_t* param,
  kaapi_perf_idset_t* storage
)
{
  const kaapi_perf_id_t casted_param =
    (kaapi_perf_id_t)(uintptr_t)param;

  switch (casted_param)
  {
    case KAAPI_PERF_ID_TASKS:
    case KAAPI_PERF_ID_STEALREQOK:
    case KAAPI_PERF_ID_STEALREQ:
    case KAAPI_PERF_ID_STEALOP:
    case KAAPI_PERF_ID_SUSPEND:
    case KAAPI_PERF_ID_T1:
  /*  case KAAPI_PERF_ID_TIDLE: */
    case KAAPI_PERF_ID_TPREEMPT:
    case KAAPI_PERF_ID_TASKLISTCALC:
    case KAAPI_PERF_ID_PAPI_0:
    case KAAPI_PERF_ID_PAPI_1:
    case KAAPI_PERF_ID_PAPI_2:
    {
      storage->count = 1;
      storage->idmap[0] = casted_param;
      param = storage;
      break;
    }

  default:
    break;
  }

  return param;
}


/*
*/
void _kaapi_perf_accum_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  unsigned int k;
  unsigned int i;
  unsigned int j;
  kaapi_perf_idset_t local_idset;
  kaapi_perf_counter_t accum[KAAPI_PERF_ID_MAX];

  kaapi_assert( (isuser ==KAAPI_PERF_USR_COUNTER) ||
                (isuser==KAAPI_PERF_SYS_COUNTER)  ||
                (isuser== (KAAPI_PERF_SYS_COUNTER | KAAPI_PERF_USR_COUNTER) )
  );

  idset = get_perf_idset(idset, &local_idset);

  memset(accum, 0, sizeof(accum));
  
  for (k = 0; k < kaapi_count_kprocessors; ++k)
  {
    const kaapi_processor_t* const kproc = kaapi_all_kprocessors[k];
    
    if (isuser ==KAAPI_PERF_USR_COUNTER)
    {
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_USER_STATE][i];
    }
    else if (isuser ==KAAPI_PERF_SYS_COUNTER)
    {
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_SCHEDULE_STATE][i];
    }
    else
      for (i = 0; i < KAAPI_PERF_ID_MAX; ++i)
        accum[i] += kproc->perf_regs[KAAPI_PERF_USER_STATE][i] + kproc->perf_regs[KAAPI_PERF_SCHEDULE_STATE][i];
  }
  
  /* filter */
  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      counter[j++] += accum[i];
}


/*
*/
void _kaapi_perf_read_counters(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_assert( (isuser ==0)||(isuser==1) );

  memset(counter, 0, idset->count * sizeof(kaapi_perf_counter_t) );
  _kaapi_perf_accum_counters( idset, isuser, counter );
}


/*
 */
void _kaapi_perf_read_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* counter)
{
  kaapi_processor_t* kproc;
  unsigned int j;
  unsigned int i;
  kaapi_perf_idset_t local_idset;

  kaapi_assert( (isuser ==0)||(isuser==1) );
  kproc = kaapi_get_current_processor();
  idset = get_perf_idset(idset, &local_idset);

  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      counter[j++] = kproc->perf_regs[isuser][i];
}


/*
 */
void _kaapi_perf_accum_register(const kaapi_perf_idset_t* idset, int isuser, kaapi_perf_counter_t* accum)
{
  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  unsigned int j;
  unsigned int i;
  kaapi_perf_idset_t local_idset;

  idset = get_perf_idset(idset, &local_idset);

  for (j = 0, i = 0; j < idset->count; ++i)
    if (idset->idmap[i])
      accum[j++] += kproc->perf_regs[isuser][i];
}


const char* kaapi_perf_id_to_name(kaapi_perf_id_t id)
{
  static const char* names[] =
  {
    "TASKS",
    "STEALREQOK",
    "STEALREQ",
    "STEALOP",
    "SUSPEND",
    "PAPI_0",
    "PAPI_1",
    "PAPI_2"
  };
  
  if (id <5) return names[(size_t)id];
  return kaapi_perf_id_to_name( id - 5 );
}


size_t kaapi_perf_counter_num(void)
{
  return kaapi_mt_perf_counter_num();
}


void kaapi_perf_idset_zero(kaapi_perf_idset_t* set)
{
  set->count = 0;
  memset(set->idmap, 0, sizeof(set->idmap));
}


void kaapi_perf_idset_add(kaapi_perf_idset_t* set, kaapi_perf_id_t id)
{
  if (set->idmap[(size_t)id])
    return ;

  set->idmap[(size_t)id] = 1;
  ++set->count;
}



