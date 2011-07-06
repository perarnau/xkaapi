/*
** kaapi_hashmap.c
** xkaapi
** 
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

int kaapi_splitter_default
(kaapi_stealcontext_t* sc, int nreq,  kaapi_request_t* req, void* p)
{
 /* default splitter */

  /* victim context */
  kaapi_splitter_context_t* const vw = (kaapi_splitter_context_t*)p;
  
  /* stolen range */
  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;

  size_t total_size;
  
  /* reply count */
  int nrep = 0;
  
  /* size per request */
  kaapi_workqueue_index_t unit_size;

redo_steal:
  /* do not steal if range size <= PAR_GRAIN */
#define CONFIG_PAR_GRAIN 4
  range_size = kaapi_workqueue_size(&vw->wq);

  if (range_size <= CONFIG_PAR_GRAIN) return 0;

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
  if (kaapi_workqueue_steal(&vw->wq, &i, &j, nreq * unit_size))
    goto redo_steal;

  total_size = offsetof(kaapi_splitter_context_t, data) + vw->data_size;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    /* thief work: not adaptive result because no preemption is used here  */
    kaapi_splitter_context_t* const tw = kaapi_reply_init_adaptive_task
      ( sc, req, vw->body, total_size, 0 );

    kaapi_workqueue_init(&tw->wq, j - unit_size, j);
    tw->body = vw->body;
    tw->data_size = vw->data_size;
    memcpy(tw->data, vw->data, vw->data_size);

    kaapi_reply_push_adaptive_task(sc, req);
  }
  
  return nrep;
}
