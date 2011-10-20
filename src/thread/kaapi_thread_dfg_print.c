/*
** kaapi_thread_print.c
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

/** Bits are SEAT
*/
static const char* tab_bit[] __attribute__((unused)) = {
  "0000",
  "0001",
  "0010",
  "0011",
  "0100",
  "0101",
  "0110",
  "0111",
  "1000",
  "1001",
  "1010",
  "111",
  "1100",
  "1101",
  "1110",
  "1111"
};

/*
 * E -> execution
 * S -> steal
 * _ -> nop 
 * A -> after steal op
 * T -> term
 * X -> term after steal
 */
typedef char state_type_t[4];
static void kaapi_getstatename( kaapi_task_t* task, state_type_t char_state )
{
  uintptr_t state = (uintptr_t)task->state;
  char_state[0] = (state & KAAPI_TASK_STATE_TERM ? 'T' : '_');
  char_state[1] = (state & KAAPI_TASK_STATE_MERGE ? 'A' : '_');
  char_state[2] = (state & KAAPI_TASK_STATE_EXEC ? 'E' : '_');
  char_state[3] = (state & KAAPI_TASK_STATE_STEAL ? 'S' : '_');
}

static char kaapi_getmodename( kaapi_access_mode_t m )
{
  switch (m) {
    case KAAPI_ACCESS_MODE_V:  return 'v';
    case KAAPI_ACCESS_MODE_R:  return 'r';
    case KAAPI_ACCESS_MODE_W:  return 'w';
    case KAAPI_ACCESS_MODE_CW: return 'c';
    case KAAPI_ACCESS_MODE_RW: return 'x';
    case KAAPI_ACCESS_MODE_SCRATCH: return 't';
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
  unsigned int i;
  size_t count_params;
  state_type_t state;

  fmt = kaapi_format_resolvebybody(body);
  
  char* sp;
  sp = task->sp;
  if (fmt ==0) return 0;

  count_params = kaapi_format_get_count_params(fmt, sp );
  kaapi_getstatename(task, state);

//  int st = kaapi_task_state2int( kaapi_task_getstate(task) );
//  fprintf( file, "@%p |%c%c%c%c|, name:%-20.20s, bit:%-4.4s, sp:%p, #p:%u\n", 
  fprintf( file, "@%p |%c%c%c%c|, name:%-20.20s, sp:%p, #p:%u\n", 
        (void*)task, 
        state[3], state[2], state[1], state[0],
        fmt->name, 
  //      ( ((st>=0) && (st<16)) ? tab_bit[st] : "<OB>" ),
        sp,
	(unsigned int)count_params );
        
  /* access mode */
  if (count_params >0)
  {
    for (i=0; i<count_params; ++i)
    {
      const kaapi_format_t* fmt_param = kaapi_format_get_fmt_param(fmt, i, sp );
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( kaapi_format_get_mode_param(fmt, i, sp));
      int cstate = KAAPI_ACCESS_IS_ONLYWRITE(m) ? 'W' : '?';

      fprintf( file, "\t\t\t [%u]%c:", i, (char)cstate );
      fputc(kaapi_getmodename(m), file );
      fputs("", file );
      if (m == KAAPI_ACCESS_MODE_V)
      {
        void* data = kaapi_format_get_data_param(fmt, i, sp );
        fprintf(file, "<%s >, @:%p=", fmt_param->name, data );
        (*fmt_param->print)(file, data );
      }
      else 
      {
        kaapi_access_t access = kaapi_format_get_access_param(fmt, i, sp );
        fprintf(file, "<%s > @:%p value=", fmt_param->name, access.data);
#if 0 /* due to invalid pointer */
        (*fmt_param->print)(file, access.data );
        if (access.version !=0)
        {
          fprintf(file, ", ver:%p value=", access.version );
          (*fmt_param->print)(file, access.version );
        }
#endif
#if 0 /* deprecated */
        if (KAAPI_ACCESS_IS_CUMULWRITE(m))
        {
          int wa = *kaapi_format_get_cwflag( fmt, i, sp);
          fprintf(file, ", cw_iswrite=%s", (wa !=0 ? "yes" : "no") );
        }
#endif
      }
      if (i < count_params-1)
      {
        fputs("\n", file );
      }
    }
  }

  fputc('\n', file );

  fflush(file);
  return 0;
}


/** 
*/
int kaapi_thread_print  ( FILE* file, kaapi_thread_context_t* thread )
{
  kaapi_frame_t* frame;
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  kaapi_task_body_t body;
  const kaapi_format_t* fmt;
  int count, iframe;

  if (thread ==0) return 0;

  fprintf(file,"Thread @:%p\n", (void*)thread );

  count = 0;

  frame    = kaapi_stack_topframe(&thread->stack);
  if (frame ==0) return 0;
  iframe   = 0;
  task_bot = kaapi_stack_bottomtask(&thread->stack);

  do 
  {
    fprintf(file, "%i: --------frame: @:%p  :: pc:%p, sp:%p, spd:%p, type: '%s'\n", 
        iframe, (void*)frame, (void*)frame->pc, (void*)frame->sp, (void*)frame->sp_data,
        (frame->tasklist == 0 ? "DFG" : "Static")
    );
    task_top = frame->sp;
    while ( task_bot > task_top)
    {
      body = kaapi_task_getbody(task_bot);
      fmt = kaapi_format_resolvebybody( body );
      
      if (fmt ==0) 
      {
        const char* fname = "<empty format>";
        if (body == kaapi_nop_body) 
          fname = "nop";
        else if (body == (kaapi_task_body_t)kaapi_taskstartup_body) 
          fname = "startup";
        else if ( body == kaapi_taskmain_body) 
          fname = "maintask";
        else if (body == kaapi_tasksteal_body) 
          fname = "steal";
        else if (body == (kaapi_task_body_t)kaapi_aftersteal_body) 
          fname = "aftersteal";
        else if (body == kaapi_taskmove_body) 
          fname = "move";
        else if (body == kaapi_taskalloc_body) 
          fname = "alloc";
          
        state_type_t state;
        kaapi_getstatename(task_bot, state);
        fprintf( file, "  [%04i]: @%p |%c%c%c%c|, name:%-20.20s", 
              count, 
              (void*)task_bot,
              state[3], state[2], state[1], state[0],
              fname
        );
        
        if (body == kaapi_tasksteal_body)
        {
          kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task_bot, kaapi_tasksteal_arg_t );
          fprintf(file, ", thief task:" );
          kaapi_task_print(file, arg->origin_task, body );
        }
        else if (body == (kaapi_task_body_t)kaapi_aftersteal_body)
        {
          fprintf(file, ", steal/term task:" );
          kaapi_task_print(file, task_bot, body );
        }
        fputc('\n', file);
        
        if ((body == kaapi_taskmove_body) || (body == kaapi_taskalloc_body))
        {
          kaapi_move_arg_t* arg = kaapi_task_getargst( task_bot, kaapi_move_arg_t );
          fprintf( file, "\t\t\t [0]?:r<____>  @:%p\n", (void*)arg->src_data.ptr.ptr );
          fprintf( file, "\t\t\t [1]?:v<view>  type:%i, size:%lu\n", arg->src_data.view.type, 
                kaapi_memory_view_size(&arg->src_data.view) );
          fprintf( file, "\t\t\t [2]?:w<____>  H@:%p\n", (void*)arg->dest );
        }
        ++count;

        --task_bot;
        continue;
      }

      /* print the task */
      fprintf( file, "  [%04i]: ", count );
      kaapi_task_print(file, task_bot, body );

      ++count;
      --task_bot;
    }
    task_bot = task_top;
    if (frame->tasklist !=0)
      kaapi_thread_tasklist_print( file, frame->tasklist );

    ++frame;
    ++iframe;
  } while (frame <= thread->stack.sfp);

  fflush(file);
  return 0;
}
