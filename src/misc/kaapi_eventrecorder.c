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

/*
*/
extern const char* get_kaapi_git_hash(void);

/* global mask of events to register */
uint64_t kaapi_event_mask;

/*
*/
uint64_t kaapi_event_startuptime = 0;

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


/** The thread to join in termination
*/
static pthread_t collector_threadid;


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
    
    /* write the header */
    kaapi_eventfile_header header;
    header.version         = __KAAPI__;
    header.minor_version   = __KAAPI_MINOR__;
    header.trace_version   = __KAAPI_TRACE_VERSION__;
    header.cpucount        = kaapi_default_param.cpucount;
    header.event_mask      = kaapi_event_mask;
    const char* gitversion = get_kaapi_git_hash();
    memset(header.package, 0, sizeof(header.package));
    strncpy(header.package, gitversion, sizeof(header.package)-1);
    ssize_t sz_write = write(listfd_set[kid], &header, sizeof(header));
    kaapi_assert(sz_write == sizeof(header));
  }
  ssize_t sz_write = write(listfd_set[kid], evb->buffer, sizeof(kaapi_event_t)*evb->pos);
  kaapi_assert( sz_write == (ssize_t)(sizeof(kaapi_event_t)*evb->pos) );
  evb->pos = 0;
}


/* Write all buffers in the list */
void kaapi_event_fencebuffers(void)
{
  kaapi_event_buffer_t* evb;

  pthread_mutex_lock(&mutex_listevt);
  while (listevt_head !=0)
  {
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
    
    pthread_mutex_lock(&mutex_listevt);
  }
  pthread_mutex_unlock(&mutex_listevt);
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
kaapi_event_buffer_t* kaapi_event_openbuffer(int kid)
{
  kaapi_event_buffer_t* evb = (kaapi_event_buffer_t*)malloc(sizeof(kaapi_event_buffer_t));
  evb->pos  = 0;
  evb->kid  = kid;
  evb->next = 0;
  return evb;
}


/**
*/
kaapi_event_buffer_t* kaapi_event_flushbuffer( kaapi_event_buffer_t* evb )
{
  if (evb ==0) return 0;

  /* push buffer in listevt buffer list */
  int kid = evb->kid;
  pthread_mutex_lock(&mutex_listevt);
  evb->next = 0;
  if (listevt_head !=0)
    listevt_tail->next = evb;
  else listevt_head = evb;
  listevt_tail = evb;
  pthread_cond_signal(&signal_thread);
  pthread_mutex_unlock(&mutex_listevt);

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
  evb->kid  = kid;

  return evb;
}


/*
*/
void kaapi_event_closebuffer( kaapi_event_buffer_t* evb )
{
  if (evb ==0) return;

  pthread_mutex_lock(&mutex_listevt);
  evb->next = 0;
  if (listevt_head !=0)
    listevt_tail->next = evb;
  else listevt_head = evb;
  listevt_tail = evb;
  pthread_cond_signal(&signal_thread);
  pthread_mutex_unlock(&mutex_listevt);
}


/**
*/
void kaapi_eventrecorder_init(void)
{
  int i;
  
  kaapi_event_startuptime = kaapi_event_date();
  
  /* mask all events types */
  kaapi_event_mask= ~0UL;
  
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    listfd_set[i] = -1;

  pthread_mutex_init(&mutex_listevt, 0);
  pthread_mutex_init(&mutex_listevtfree_head, 0);
  pthread_cond_init(&signal_thread, 0);
  pthread_create(&collector_threadid, 0, _kaapi_event_flushimator, 0);
}


/** Finish trace. Assume that threads have reach the barrier and have flush theirs buffers
*/
void kaapi_eventrecorder_fini(void)
{
  void* result;
  int i;
  kaapi_event_buffer_t* evb;

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
    
  /* close all file descriptors */
  for (i=0; i<KAAPI_MAX_PROCESSOR; ++i)
    if (listfd_set[i] != -1)
      close(listfd_set[i]);

  /* destroy mutexes/conditions */
  pthread_cond_destroy(&signal_thread);
  pthread_mutex_destroy(&mutex_listevt);
  pthread_mutex_destroy(&mutex_listevtfree_head);
}
