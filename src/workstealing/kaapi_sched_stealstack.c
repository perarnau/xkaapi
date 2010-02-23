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

/* fwd decl */
static int kaapi_updateaccess_ready( 
  kaapi_hashmap_t* access_to_gd,
  kaapi_task_t* task_bot, 
  const kaapi_format_t* task_fmt, 
  kaapi_task_state_t state 
);

static int kaapi_search_framebound( kaapi_task_t* beg, kaapi_task_t** end, kaapi_task_t** curr, kaapi_task_t* endmax );

static int kaapi_update_version( 
  kaapi_hashmap_t* access_to_gd,
  int count, 
  kaapi_task_t* beg, 
  kaapi_task_t* endmax 
);

/*
*/
static int kaapi_updateaccess_ready( 
  kaapi_hashmap_t* access_to_gd, 
  kaapi_task_t* task, 
  const kaapi_format_t* fmt, 
  kaapi_task_state_t state 
)
{
  int i;
  int countparam;
  int waitparam;
  
  /* if access INIT & flag == READY, do not recompute readiness, else update version */
  if ((task->flag & KAAPI_TASK_MASK_READY) && (state ==KAAPI_TASK_S_INIT)) return 0;
  
  /* task without synchronisation */
  if (!kaapi_task_issync( task )) 
  {
    if (kaapi_task_isadaptive(task) && ((state ==KAAPI_TASK_S_EXEC) || (state == KAAPI_TASK_S_INIT)))
    {
      task->flag |= KAAPI_TASK_MASK_READY;
      return 0;
    }
    task->flag &= ~KAAPI_TASK_MASK_READY;
    return 0;
  }

  countparam = waitparam = fmt->count_params;
  for (i=0; i<countparam; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    if (m == KAAPI_ACCESS_MODE_V) 
    {
      --waitparam;
      continue;
    }
    kaapi_access_t* access = (kaapi_access_t*)(fmt->off_params[i] + (char*)kaapi_task_getargs(task));

    /* key point here to be able to reuse user data */
    kaapi_gd_t* gd = &kaapi_hashmap_find( access_to_gd, access->data )->value;
    
    switch (state) {
      case KAAPI_TASK_S_INIT:
        if ((gd->last_mode == KAAPI_ACCESS_MODE_VOID) || KAAPI_ACCESS_IS_CONCURRENT(m, gd->last_mode))
        {
          gd->last_version = access->data;
          access->version = gd->last_version;
          --waitparam; 
        }
        else 
        {
          gd->last_version = 0;
          access->version  = 0;
        }
        gd->last_mode = m;
      break;

      case KAAPI_TASK_S_TERM:
          gd->last_version = access->data;          /* this is the data */
          gd->last_mode = KAAPI_ACCESS_MODE_R; /* and it could be read */
          --waitparam;
      break;

      case KAAPI_TASK_S_WAIT:
      case KAAPI_TASK_S_EXEC:
      case KAAPI_TASK_S_STEAL:
      {
        int r1 __attribute__((unused)) = KAAPI_ACCESS_IS_ONLYWRITE(m);
        int r2 __attribute__((unused)) = KAAPI_ACCESS_IS_READWRITE(m);
        if (KAAPI_ACCESS_IS_ONLYWRITE(m) || KAAPI_ACCESS_IS_READWRITE(m))
          gd->last_version = 0;              /* data not produced */
        else {
          gd->last_version = access->data;   /* data R -> is ready */
        }
        gd->last_mode = m;
        --waitparam; /* but parameter is ready ! */
      } break;
    }
  }
  if (waitparam ==0)
    task->flag |= KAAPI_TASK_MASK_READY;
  else 
    task->flag &= ~KAAPI_TASK_MASK_READY;
  return waitparam;
}


/* Warning: frame are [beg, end] with beg >= end because stack of task growth down.
*/
static int kaapi_search_framebound( kaapi_task_t* beg, kaapi_task_t** end, kaapi_task_t** curr, kaapi_task_t* endmax)
{
  while (beg != endmax)
  {
    if (kaapi_task_getbody(beg) == kaapi_retn_body)
    {
      kaapi_frame_t* frame = kaapi_task_getargst( beg, kaapi_frame_t );
      *end = beg;
      *curr = frame->sp;
      return 1;
    }
    --beg;
  }
  *curr = 0;
  *end = endmax;
  return 0;
}


/* update all versions & readiness flag for tasks in the frame [beg, end(, curr is the current task of the frame 
*/
static int kaapi_update_version( kaapi_hashmap_t* access_to_gd, int count, kaapi_task_t* beg, kaapi_task_t* endmax )
{
  const kaapi_format_t* fmt;
  int waitparam;
  kaapi_task_state_t state;
  kaapi_task_t* end;
  kaapi_task_t* curr =0;
  
  /* search frame bound */
  kaapi_search_framebound( beg, &end, &curr, endmax );

  kaapi_assert_debug( (end ==0) || (end == endmax) || (kaapi_task_getbody(end) == kaapi_retn_body) );
#if 0
  printf("Update frame: beg:%p  - end:%p, pc: %p\n", (void*)beg, (void*)end, (void*)curr );
#endif

  while ((beg != endmax) && (beg != end) && (count >0))
  {
    fmt = kaapi_format_resolvebybody( kaapi_task_getbody(beg) );

    if (fmt ==0) 
    {
      --beg;
      continue;
    }

    state = kaapi_task_getstate( beg );
    waitparam = kaapi_updateaccess_ready( access_to_gd, beg, fmt, state );

    /* TODO: Optimization: can i steal it ? to decr count and return quickly */
    if (waitparam ==0) 
    {
      if (kaapi_task_isadaptive(beg) && (state == KAAPI_TASK_S_EXEC)) return count = 0; 
      if ((kaapi_task_isstealable(beg)||kaapi_task_isadaptive(beg))&& (state == KAAPI_TASK_S_INIT))
      {
//printf("Task: %p isready, count:%i\n", (void*)beg,count);
        --count;
return 0;
      }
#if 0
      /* found ready task*/
      if (count ==0) 
      {
#if 0
        printf("Abort update version on Task: %p \n", (void*)beg);
#endif
        return 0;
      }
#endif
    }

    --beg;
  }

  /* find in sub frame */
  if (curr !=0) 
  {
    kaapi_assert_debug( (end !=0) && (kaapi_task_getbody(end) == kaapi_retn_body) );
    if (end-1 > endmax) 
    { /* recursive call on sub frame */
      kaapi_update_version( access_to_gd, count, end-1, endmax );
    }
  }
  
  return count;
}


/** Steal task in the stack from the bottom to the top.
    Do not steal curr if !=0 (current running adaptive task) in case of cooperative WS.
*/
int kaapi_sched_stealstack  ( kaapi_stack_t* stack, kaapi_task_t* curr )
{
  kaapi_task_t*         task_top;
  kaapi_task_t*         task_bot;
  const kaapi_format_t* task_fmt;
  int count, savecount;
  int step;
  int replycount;
  int isready;
  int isupdateversion = 0;
  kaapi_task_state_t state;
  
  kaapi_hashmap_t access_to_gd;
/*  kaapi_hashentries_bloc_t stackbloc; */

#if defined(KAAPI_CONCURRENT_WS)
  count = KAAPI_ATOMIC_READ(&stack->_proc->hlrequests.count);
#else
  count = stack->hasrequest;
#endif
  if (count ==0) return 0;
  if (kaapi_stack_isempty( stack)) return 0;


  kaapi_hashmap_init( &access_to_gd, 0 ); //&stackbloc );

  savecount = count;

#if 0
printf("------ STEAL STACK @:%p\n", (void*)stack );
//kaapi_stack_print(0, stack);
#endif

#if 0 /*defined(KAAPI_USE_PERFCOUNTER)*/
  double t1, t0 = kaapi_get_elapsedtime();
#endif
  /* reset dfg constraints evaluation */
  step = 0;
  
 redo_compute: 
  /* iterate through all the tasks from task_bot until task_top */
  task_bot = kaapi_stack_bottomtask(stack);
  task_top = kaapi_stack_toptask(stack);

  replycount = 0;

  while ((count >0) && (task_bot !=0) && (task_bot != task_top))
  {
    if (task_bot == 0) break;

    task_fmt = 0;
    if (!kaapi_task_isadaptive(task_bot))
      task_fmt = kaapi_format_resolvebybody( kaapi_task_getbody(task_bot) );
    if ((task_fmt ==0) && (!kaapi_task_isadaptive(task_bot)))
    {
      --task_bot;
      continue;
    }

    /* compute dependency */
    if (step == 0)
    {
      if ( (isupdateversion ==0) && kaapi_task_issync(task_bot) )
      {
        kaapi_update_version( &access_to_gd, count, task_bot, task_top );
#if 0
        printf("================ AFTER UPDATE VERSION \n");
        fflush(stdout);
        kaapi_stack_print( 1, stack);
        printf("================  \n");
        fflush(stdout);
#endif
        isupdateversion = 1;
      }
    }
    
    /* */
    isready = kaapi_task_isready(task_bot);
#if defined(KAAPI_CONCURRENT_WS)
    if ((!isready) && kaapi_task_issync(task_bot))
#else
    if ((!isready || ((curr !=0) && (task_bot == curr))) && kaapi_task_issync(task_bot))
#endif
    {
      /* next task */
      --task_bot;
      continue;
    }

    state = kaapi_task_getstate( task_bot );
    if ((state == KAAPI_TASK_S_TERM) || (state == KAAPI_TASK_S_STEAL) || !kaapi_task_isstealable(task_bot))
    {
      /* next task */
      --task_bot;
      continue;
    }

    /* Is it a DFG or task that will require or introduce synchronisations ? */
    if (kaapi_task_issync(task_bot) && (state == KAAPI_TASK_S_INIT))
    {
      int retval = kaapi_task_splitter_dfg(stack, task_bot, count, stack->requests );      
#if 0/*defined(KAAPI_USE_PERFCOUNTER)*/
      t1 = kaapi_get_elapsedtime();
      printf("[STEAL] date:%15f, delay: %f\n", t0, t1-t0);
#endif
      count -= retval;
      replycount += retval;
      kaapi_assert_debug( count >=0 );
    }
    else if (kaapi_task_isadaptive(task_bot)) 
    {
      kaapi_taskadaptive_t* ta = (kaapi_taskadaptive_t*)task_bot->sp;
      if ((state == KAAPI_TASK_S_INIT) && (task_fmt !=0)) /* steal the entire task (always better !) */
      {
        int retval = kaapi_task_splitter_dfg(stack, task_bot, count, stack->requests );      
#if 0/*defined(KAAPI_USE_PERFCOUNTER)*/
        t1 = kaapi_get_elapsedtime();
        printf("[STEAL] date:%15f, delay: %f\n", t0, t1-t0);
#endif
        count -= retval;
        replycount += retval;
        kaapi_assert_debug( count >=0 );
      }
      else if ( (ta->splitter !=0) && ((state == KAAPI_TASK_S_EXEC) || (state == KAAPI_TASK_S_INIT)) ) /* partial steal */
      {
        int retval = (*ta->splitter)(stack, task_bot, count, stack->requests, ta->argsplitter );
        count -= retval;
        /* never increment replycount here, because user splitter function was call and has already
           decrement the request counter
           replycount += retval;
        */
        kaapi_assert_debug( count >=0 );
      }
    }
    
    --task_bot;
  }

#if 0
printf("------ END STEAL @:%p\n", (void*)stack );
#endif
  
  kaapi_hashmap_destroy( &access_to_gd );

  if (replycount >0)
  {
    KAAPI_ATOMIC_SUB( &stack->_proc->hlrequests.count, replycount );
  }

  return savecount-count;
}
