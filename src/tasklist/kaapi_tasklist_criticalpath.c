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


static inline uint64_t _kaapi_max(uint64_t d1, uint64_t d2)
{ return (d1 < d2 ? d2 : d1); }



/* explore successor in recursive maner 
*/
static void kaapi_explore_successor( 
  kaapi_taskdescr_t*      td
)
{
  uint64_t                maxdatesucc = 0;
  kaapi_activationlink_t* curr;

  /* if already visited, return */
  if (td->mark) 
    return ;
    
  curr = td->u.acl.list.front;
  if (curr == 0) 
  {
    /* terminal case: */
    td->mark = 1;
    td->u.acl.date = 0;
    return ;
  }

  /* iterate of all successor and get the maximum value */
  while (curr !=0)
  {
    if (curr->td->mark ==0) 
      kaapi_explore_successor(curr->td);
    maxdatesucc = _kaapi_max(maxdatesucc, curr->td->u.acl.date);

    curr = curr->next;
  }
  td->mark = 1;
  td->u.acl.date = maxdatesucc+1;
}


/* compute the critical path of each task : length to the final execution execution
*/
int kaapi_tasklist_critical_path( kaapi_frame_tasklist_t* tasklist )
{
  kaapi_taskdescr_t*         td;         

  if (tasklist == 0)
    return EINVAL;

  /* iterate over all tasks and:
     - associated a td in task2task_khm were list points to the list of predecessors
     - populate initial tasks with no sucessor 
  */
  kaapi_activationlink_t* curr = tasklist->readylist.front;
  while (curr !=0)
  {
    td = curr->td;
    /* mark and explore all successors of td: on return td->date was computed */
    kaapi_explore_successor( td );
    
    /* */
    tasklist->t_infinity = _kaapi_max( tasklist->t_infinity, td->u.acl.date);
    curr = curr->next;
  }

  return 0;  
}
