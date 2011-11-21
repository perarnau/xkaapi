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

/** */
int kaapi_workqueue_slowpop(
  kaapi_workqueue_t* kwq, 
  kaapi_workqueue_index_t* beg,
  kaapi_workqueue_index_t* end,
  kaapi_workqueue_index_t max_size
)
{
  kaapi_workqueue_index_t size;
  kaapi_workqueue_index_t loc_beg;

  if ( kwq->lock ==0 ) return ESRCH;

  kaapi_sched_lock( kwq->lock );

  loc_beg = kwq->beg;
  size = kwq->end - loc_beg;
  if (size ==0) 
    goto empty_case;
  if (size > max_size)
    size = max_size;
  loc_beg += size;
  kwq->beg = loc_beg;

  kaapi_mem_barrier();
  kaapi_sched_unlock( kwq->lock );

  *end = loc_beg;
  *beg = *end - size;
  return 0;

empty_case:
  kaapi_sched_unlock( kwq->lock );
  return EBUSY;
}
