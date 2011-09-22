/*
** kaapi_hws_pushtask.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
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
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"

/* push a task at a given hierarchy level
 */

int kaapi_hws_pushtask
(kaapi_task_body_t body, void* data, kaapi_hws_levelid_t levelid)
{
  /* kaapi_assert(kaapi_hws_is_levelid_set(levelid)); */

  kaapi_processor_t* const kproc = kaapi_get_current_processor();
  kaapi_ws_block_t* const block = hws_levels[levelid].kid_to_block[kproc->kid];
  kaapi_ws_queue_t* const queue = block->queue;

  /* toremove */
  kaapi_hws_sched_inc_sync();
  /* toremove */

  kaapi_ws_queue_push(queue, body, data);

  return 0;
}
