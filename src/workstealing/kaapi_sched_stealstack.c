/*
** kaapi_sched_stealstack.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributor :
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

/* Compute if the task with arguments pointed by sp and with format task_fmt is ready
   Return the number of non ready data
*/
static int kaapi_task_computeready( void* sp, const kaapi_format_t* task_fmt, kaapi_hashmap_t* map )
{
  int i, wc, countparam;
  countparam = wc = task_fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(task_fmt->mode_params[i]);
    if (m == KAAPI_ACCESS_MODE_V) 
    {
      --wc;
      continue;
    }
    kaapi_access_t* access = (kaapi_access_t*)(task_fmt->off_params[i] + (char*)sp);

    /* */
    kaapi_gd_t* gd = &kaapi_hashmap_find( map, access->data )->value;
    
    /* compute readyness of access */
    if ( KAAPI_ACCESS_IS_ONLYWRITE(m)
      || (gd->last_mode == KAAPI_ACCESS_MODE_VOID)
      || (KAAPI_ACCESS_IS_CONCURRENT(m, gd->last_mode))
       )
    {
      --wc;
    }
    
    /* update map information for next access if no set */
    if (gd->last_mode == KAAPI_ACCESS_MODE_VOID)
      gd->last_mode = m;
    
    /* currently, datas produced by aftersteal_task are visible to thief in order to augment
       the parallelism by breaking chain of versions (W->R -> W->R ), the second W->R could
       be used (the middle R->W is splitted -renaming is also used in other context-).
       But we do not take into account of this extra parallelism.
       
       The problem that remains to solve:
        - if task_bot is kaapi_aftersteal_body task, then it corresponds to a stolen task already
        finished (before the victim thread is trying to execute it, else we will see an nop_body. 
        In this task:
          - each access->data points to the original data in the victim thread
          - each access->version points either to 0 (no new data produced) either to an heap allocated
          data value != access->data that may be used for next reader.
          This situation is only all the case W, CW access mode accesses.
        - this task_bot is considered as terminated and W, CW or RW data value 'access->version' may be consumed but
          - access->version (for W or CW) may be released if the victim thread executes the task 'aftersteal'
          - in the same time, this pointed data may be read by the stolen closure (if all its access are ready, 
          here we do not know about this fact: we need to wait after looking for all others parameters of the current task)

        A solution is to store references to accesses to last version in gd table:
          - if a closure is found to be ready:
              - we assume than gd map will store a pointer to the version to read (or write).
              - the victim will made a call to 'cas(access->version, access->version, 0)' to delete version
                  1/ if ok -> the victim copy the data into the gd an set to 0 its version
                  2/ if nok -> the victim only copy the version to the gd, 2 versions of the data will 
                  be alive until the next aftersteal (that cannot be executed because the victim if currently
                  under executing a first (previous) after steal).
              - the thief will made a call to 'cas(gd->access->version, gd->version, 0)' to keep the owner ship 
              on the data gd->access->version.
                  1/ if ok -> the thief get the owner ship of data which will be deleted during the aftersteal
                  of the under stealing closure
                  2/ ifnok -> the victim has executed the aftersteal, then the victim may read gd->data as the correct version
                  (this guarantee has to be written to ensure than reading gd->data is yet valid)
                  (no other tasks have modified the shared between the 1rst aftersteal the task detected to be stolen)
            - seems good... with more details
    */
  }
  return wc;
}


/** Steal task in the frame that begin at bot until retn
    - precond: the frame must be locked by setting to nop the task that has created the frame
    - poscond: the frame is locked
*/
static int kaapi_sched_stealframe(
    kaapi_stack_t* stack, kaapi_task_t* task_bot, 
    kaapi_hashmap_t* map, 
    int count, kaapi_request_t* request 
)
{
  const kaapi_format_t* task_fmt;
  kaapi_task_body_t     task_body;
  kaapi_task_t*         task_top;
  kaapi_task_t*         task_exec;
  int                   replycount;

  /* suppress history of the previous frame ! */
  kaapi_hashmap_clear( map );
  
  task_body  = kaapi_nop_body;
  task_top   = kaapi_stack_toptask(stack);
  task_exec  = 0;
  replycount = 0;
  while ( (count > replycount) && (task_bot != task_top) )
  {
    task_body = kaapi_task_getbody(task_bot);
    /* skip nop: nop correspond to already executed task, do not find in hash map
       their data because the next lookup will return data as "not known" and thus ready whatever is
       the access mode.
    */
    if (task_body == kaapi_nop_body) 
      goto label_continue;

    if (task_body == kaapi_exec_body)
      task_exec = task_bot;
    task_fmt = kaapi_format_resolvebybody( kaapi_task_getextrabody(task_bot) );
    if (task_fmt !=0)
    {
      int wc = kaapi_task_computeready( kaapi_task_getargs( task_bot), task_fmt, map );
      if ((wc ==0) && kaapi_task_isstealable(task_bot))
      {
        task_body = kaapi_task_getextrabody(task_bot);
        if (kaapi_task_casstate(task_bot, task_body, kaapi_suspend_body)) 
        {
          replycount += kaapi_task_splitter_dfg(stack, task_bot, count-replycount, stack->requests );
        }
      }
    }
label_continue:
    --task_bot;
  }
  if (replycount == count) return replycount;
  kaapi_assert_debug( replycount <= count );

  /* recursive call */
  if ((task_body == kaapi_retn_body) && (task_exec !=0))
  {
    /* lock the subframe first by locking the task_exec (its kaapi_exec_body) so that retn cannot pop task */
    if (kaapi_task_casstate(task_exec, kaapi_exec_body, kaapi_nop_body)) 
    {
      replycount += kaapi_sched_stealframe( stack, task_bot-1, map, count-replycount, request );
      /* reset normal body on task_exec */
      kaapi_task_setbody(task_exec, kaapi_exec_body);
    }
  }
  return replycount;
}


/** Steal task in the stack from the bottom to the top.

    Do not steal curr if !=0 (current running adaptive task) in case of cooperative WS.
    This signature is the same as a splitter function.
*/
int kaapi_sched_stealstack  ( kaapi_stack_t* stack, kaapi_task_t* curr, int count, kaapi_request_t* request )
{
  kaapi_task_t*            task_bot;
  kaapi_task_body_t        task_body;
  int savecount;
  int replycount;
  
  kaapi_hashmap_t          access_to_gd;
  kaapi_hashentries_bloc_t stackbloc;

  if (count ==0) return 0;
  if (kaapi_stack_isempty( stack)) return 0;

  savecount = count;

  /* be carrefull, the map should be clear before used */
  kaapi_hashmap_init( &access_to_gd, &stackbloc );

  
  /* steal frame by frame in recursive fashion */
  task_bot  = kaapi_stack_bottomtask(stack);
  task_body = kaapi_task_getbody(task_bot);

  /* assume task_body == kaapi_exec_body, lock it in order to avoid the retn */
  if (!kaapi_task_casstate(task_bot, kaapi_exec_body, kaapi_nop_body)) return 0;
  replycount = kaapi_sched_stealframe( stack, task_bot, &access_to_gd, count, request );
  kaapi_task_setbody(task_bot, task_body);

  kaapi_hashmap_destroy( &access_to_gd );

  return replycount;
}


/*
*/
int kaapi_sched_stealstack_helper( kaapi_stack_t* stack, kaapi_task_t* curr )
{
  return kaapi_sched_stealstack( stack, curr, stack->hasrequest, stack->requests); 
}
