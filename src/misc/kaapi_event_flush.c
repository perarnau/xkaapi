/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "kaapi_impl.h"
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/**
*/
void kaapi_event_flushbuffer( kaapi_processor_t* kproc )
{
  kaapi_event_buffer_t* evb = kproc->eventbuffer;
  if (kproc->eventbuffer ==0) return;
  
  /* open if not yet opened */
  if (evb->fd == -1)
  {
    char filename[128]; 
    if (getenv("USER") !=0)
      sprintf(filename,"/tmp/events.%s.%i.evt", getenv("USER"), kproc->kid );
    else
      sprintf(filename,"/tmp/events.%i.evt", kproc->kid );

    /* open it */
    evb->fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC);
    kaapi_assert( evb->fd != -1 );
    fchmod( evb->fd, S_IRUSR|S_IWUSR);
  }
  
  /* write the buffer [0, pos) */
  ssize_t sz_write = write(evb->fd, evb->buffer, sizeof(kaapi_event_t)*evb->pos);
  kaapi_assert( sz_write == (ssize_t)(sizeof(kaapi_event_t)*evb->pos) );
  evb->pos = 0;
}


/*
*/
void kaapi_event_closebuffer( kaapi_processor_t* kproc )
{
  kaapi_event_buffer_t* evb = kproc->eventbuffer;
  if (evb ==0) return;

  kaapi_event_flushbuffer(kproc);
  if (evb->fd != -1) close(evb->fd);
  evb->fd = -1;
}


/**
*/
void _kaapi_signal_dump_counters(int xxdummy)
{
  for (uint32_t i=0; i<kaapi_count_kprocessors; ++i)
  {
    kaapi_event_closebuffer( kaapi_all_kprocessors[i] );
  }
  _exit(0);
}

