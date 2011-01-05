/*
** xkaapi
** 
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
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
#include "kaapi_impl.h"


/* ready list routines
 */
kaapi_thread_context_t* kaapi_sched_stealready(kaapi_processor_t* kproc, kaapi_processor_id_t thief_kid)
{
  kaapi_lready_t* const list = &kproc->lready;

  kaapi_thread_context_t* pos;
  kaapi_thread_context_t* pre;

  if (list->_front == 0)
    return 0;

  pre = list->_front;
  if (kaapi_cpuset_has(pre->affinity, thief_kid))
  {
    /* unlink the thread */
    pos = list->_front;
    list->_front = pos->_next;
    if (list->_back == pos) list->_back = 0;
    return pos;
  }

  pos = pre->_next;
  while (pos != 0) 
  {
    if (kaapi_cpuset_has(pos->affinity, thief_kid))
    {
      /* unlink the thread */
      pre->_next = pos->_next;
      if (pre->_next ==0) list->_back = pre;
      return pos;
    }

    pre = pos;
    pos = pos->_next;
  }

  /* todo: should return 0 (?) */
  return pos;
}


/*
*/
void kaapi_sched_pushready(kaapi_processor_t* kproc, kaapi_thread_context_t* node)
{
  /* push back version. prev not updated.
   */
  kaapi_lready_t* const list = &kproc->lready;

  node->_next = 0;
#if defined(KAAPI_DEBUG)
  node->_prev = 0;
#endif

  if (list->_back != 0)
    list->_back->_next = node;
  else
    list->_front = node;

  list->_back = node;
}
