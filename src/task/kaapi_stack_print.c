/*
** kaapi_stack_print.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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
#include <stdio.h>

/*
 * E -> execution
 * S -> steal
 * T -> terminee ou nop
 * X -> terminee apres vol
 */
static char kaapi_getstatename( kaapi_task_t* task )
{
  kaapi_task_body_t body = kaapi_task_getbody(task);
  if (body == kaapi_exec_body) return 'E';
  else if (body == kaapi_suspend_body) return 'S';
  else if (body ==kaapi_nop_body) return 'T';
  else if (body ==kaapi_aftersteal_body) return 'X';
  return 'I';
}

static char kaapi_getmodename( kaapi_access_mode_t m )
{
  switch (m) {
    case KAAPI_ACCESS_MODE_V:  return 'v';
    case KAAPI_ACCESS_MODE_R:  return 'r';
    case KAAPI_ACCESS_MODE_W:  return 'w';
    case KAAPI_ACCESS_MODE_CW: return 'c';
    case KAAPI_ACCESS_MODE_RW: return 'x';
    default: return '!';
  }
}


/**
*/
int kaapi_task_print( 
  FILE* file,
  kaapi_task_t* task, 
  kaapi_task_body_t body 
)
{
  const kaapi_format_t* fmt;
  int i;

  fmt = kaapi_format_resolvebybody( body );
  if (fmt ==0) return 0;

  fprintf( file, "@%p |%c|, name:%-15.15s, sp:%p, #p:%i ", 
        (void*)task, 
        kaapi_getstatename(task), 
        fmt->name, 
        task->sp,
        fmt->count_params );
        
  /* access mode */
  if (fmt->count_params >0)
  {
    fputs( ", mode:", file );
    for (i=0; i<fmt->count_params; ++i)
    {
      const kaapi_format_t* fmt_param = fmt->fmt_params[i];
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
      fputc(kaapi_getmodename(m), file );
      fputs("", file );
      if (m == KAAPI_ACCESS_MODE_V)
      {
        void* data = (void*)(fmt->off_params[i] + (char*)kaapi_task_getargs(task));
        fprintf(file, "<%s>, @:%p=", fmt_param->name, data );
        (*fmt_param->print)(file, data );
      }
      else 
      {
        kaapi_access_t* access = (kaapi_access_t*)(fmt->off_params[i] + (char*)kaapi_task_getargs(task));
        fprintf(file, "<%s> @:%p value=", fmt_param->name, access->data);
        (*fmt_param->print)(file, access->data );
        if (access->version !=0)
        {
          fprintf(file, ", ver:%p value=", access->version );
          (*fmt_param->print)(file, access->version );
        }
      }
      if (i <fmt->count_params-1)
      {
        fputs("|| ", file );
      }
    }
  }
  fputc('\n', file );

  fflush(file);
  return 0;
}


/** 
*/
int kaapi_stack_print  ( FILE* file, kaapi_thread_context_t* thread )
{
  kaapi_frame_t* frame;
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  const kaapi_format_t* fmt;
  int count, iframe;

  if (thread ==0) return 0;

  fprintf(file,"Thread @:%p\n", (void*)thread );

  count = 0;

  frame    = thread->stackframe;
  iframe   = 0;
  task_bot = kaapi_thread_bottomtask(thread);

  do 
  {
    fprintf(file, "%i: --------frame:: pc:%p, sp:%p, spd:%p\n", iframe, (void*)frame->pc, (void*)frame->sp, (void*)frame->sp_data );
    task_top = frame->sp;
    while (task_bot != task_top)
    {
      fmt = kaapi_format_resolvebybody( kaapi_task_getextrabody(task_bot) );
      
      if (fmt ==0) 
      {
        const char* fname = "<empty format>";
        if ( kaapi_task_getextrabody(task_bot) == kaapi_nop_body) 
          fname = "nop";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_taskstartup_body) 
          fname = "startup";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_taskstartup_body) 
          fname = "exec";
  /*
        else if ( kaapi_task_getbody(task_bot) == kaapi_retn_body) 
          fname = "retn";
  */
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_suspend_body) 
          fname = "suspend";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_taskwrite_body) 
          fname = "write";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_tasksig_body) 
          fname = "signal";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_tasksteal_body) 
          fname = "steal";
        else if ( kaapi_task_getextrabody(task_bot) == kaapi_aftersteal_body) 
          fname = "aftersteal";
          
        fprintf( file, "  [%04i]: @%p |%c|, name:%-15.15s", 
              count, 
              (void*)task_bot,
              kaapi_getstatename(task_bot), 
              fname
        );
        
        if (kaapi_task_getextrabody(task_bot) == kaapi_tasksteal_body)
        {
          kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task_bot, kaapi_tasksteal_arg_t );
          fprintf(file, ", thief task:" );
          kaapi_task_print(file, arg->origin_task, kaapi_task_getextrabody(arg->origin_task));
        }
        else if (kaapi_task_getextrabody(task_bot) == kaapi_aftersteal_body)
        {
          fprintf(file, ", steal/term task:" );
          kaapi_task_print(file, task_bot, kaapi_task_getextrabody(task_bot));
        }
        else if (kaapi_task_getextrabody(task_bot) == kaapi_suspend_body)
        {
          fprintf(file, ", steal/under task:" );
          kaapi_task_print(file, task_bot, kaapi_task_getextrabody(task_bot));
        }
        fputc('\n', file);
        ++count;

        --task_bot;
        continue;
      }

      /* print the task */
      fprintf( file, "  [%04i]: ", count );
      kaapi_task_print(file, task_bot, kaapi_task_getextrabody(task_bot) );

      ++count;
      --task_bot;
    }
    task_bot = task_top;
    ++frame;
    ++iframe;
  } while (frame <= thread->sfp);

  fflush(file);
  return 0;
}
