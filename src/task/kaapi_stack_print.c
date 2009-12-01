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


/* return the string that decode the flags */
static const char* FLAG_NAME[KAAPI_TASK_MASK_FLAGS+1];
static const char* kaapi_getflagsname( unsigned int flag )
{
  int i;
  char* buffer;
  static int isinit= 0;
  if (isinit) return FLAG_NAME[flag & KAAPI_TASK_MASK_FLAGS];
  
  buffer = malloc( 5 * (KAAPI_TASK_MASK_FLAGS+1) );
  for (i=0; i<KAAPI_TASK_MASK_FLAGS+1; ++i) 
  {
    char* name = buffer + 5*i; 
    if (i & KAAPI_TASK_STICKY) name[0] = 's';
    else name[0]='_';
    if (i & KAAPI_TASK_ADAPTIVE) name[1] = 'a';
    else name[1]='_';
    if (i & KAAPI_TASK_DFG) name[2] = 'd';
    else name[2]='_';
    if (i & KAAPI_TASK_LOCALITY) name[3] = 'l';
    else name[3]='_';
    name[4] = 0; /* end of name */
    FLAG_NAME[i] = name;
  }
  isinit = 1;
  return FLAG_NAME[flag & KAAPI_TASK_MASK_FLAGS];
}


static char kaapi_getstatename( kaapi_task_t* task )
{
  kaapi_task_state_t state = kaapi_task_getstate(task);
  switch (state) {
    case KAAPI_TASK_S_INIT:  return 'I';
    case KAAPI_TASK_S_EXEC:  return 'E';
    case KAAPI_TASK_S_STEAL: 
      if (task->body == &kaapi_aftersteal_body) 
        return 'M';
      else 
        return 'S';
    case KAAPI_TASK_S_TERM:  return 'T';
    default: return '!';
  }
}

static char kaapi_getreadyname(kaapi_task_t* task)
{
  if (task->flag & KAAPI_TASK_MASK_READY) return 'R';
  return '?';
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
int kaapi_task_print( FILE* file, kaapi_task_t* task )
{
  const kaapi_format_t* fmt;
  int i;

  if (task->format ==0) fmt = kaapi_format_resolvebybody( task->body );
  else fmt = task->format;

  fprintf( file, "@%p |%c%c%s|, name:%s, splitter:%p, #p:%i, mode: ", 
        (void*)task, 
        kaapi_getreadyname(task),
        kaapi_getstatename(task), 
        kaapi_getflagsname(task->flag), 
        fmt->name, 
	(void*)(uintptr_t)task->splitter,
        fmt->count_params );
        
  /* access mode */
  for (i=0; i<fmt->count_params; ++i)
  {
    const kaapi_format_t* fmt_param = fmt->fmt_params[i];
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);
    fputc(kaapi_getmodename(m), file );
    if (m == KAAPI_ACCESS_MODE_V)
    {
      void* data = (void*)(fmt->off_params[i] + (char*)task->sp);
      fprintf(file, "<%s>, @:%p=", fmt_param->name, data );
      (*fmt_param->print)(file, data );
    }
    else 
    {
      kaapi_access_t* access = (kaapi_access_t*)(fmt->off_params[i] + (char*)task->sp);
      kaapi_gd_t* gd = ((kaapi_gd_t*)access->data)-1;
      fprintf(file, "<%s>, @:%p, a:%p", fmt_param->name, (void*)gd, access->data );
      if (KAAPI_ACCESS_IS_READ(m) && (access->version !=0))
      {
        fputc('=', file);
        (*fmt_param->print)(file, access->version);
      }
    }
    if (i <fmt->count_params-1)
    {
      fputs("|| ", file );
    }
    else
      fputc('\n', file );
  }

  fflush(file);
  return 0;
}


/** 
*/
int kaapi_stack_print  ( int fd, kaapi_stack_t* stack )
{
  FILE* file;
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  const kaapi_format_t* fmt;
  int count;

  if (stack ==0) return 0;

  file = fdopen(fd, "a");
  if (file == 0) return EINVAL;
  
  fprintf(file,"Stack @:%p\n", (void*)stack );
  if (kaapi_stack_isempty( stack)) return 0;

  /* iterate through all the tasks from task_bot until task_top */
  task_bot = kaapi_stack_bottomtask(stack);
  task_top = kaapi_stack_toptask(stack);

  count = 0;
  while (task_bot != task_top)
  {
    if (task_bot->format ==0) fmt = kaapi_format_resolvebybody( task_bot->body );
    else fmt = task_bot->format;
    
    if (fmt ==0) 
    {
      const char* fname = "<empty format>";
      if (task_bot->body == &kaapi_retn_body) 
        fname = "retn";
      if (task_bot->body == &kaapi_suspend_body) 
        fname = "suspend";
      if (task_bot->body == &kaapi_taskwrite_body) 
        fname = "write";
      else if (task_bot->body == &kaapi_tasksig_body) 
        fname = "signal";
      else if (task_bot->body == &kaapi_tasksteal_body) 
        fname = "steal";
        
      fprintf( file, "  [%04i]: @%p |%c%c%s|, name:%s, splitter:%p", 
            count, 
            (void*)task_bot,
            kaapi_getreadyname(task_bot),
            kaapi_getstatename(task_bot), 
            kaapi_getflagsname(task_bot->flag), 
            fname, 
	    (void*)(uintptr_t)task_bot->splitter );
      
      if (task_bot->body == &kaapi_retn_body)
      {
        kaapi_frame_t* frame = kaapi_task_getargst( task_bot, kaapi_frame_t );
        fprintf(file, ", pc:%p, sp:%p, spd:%p", (void*)frame->pc, (void*)frame->sp, frame->sp_data );
      }
      else if (task_bot->body == &kaapi_tasksteal_body)
      {
        kaapi_tasksteal_arg_t* arg = kaapi_task_getargst( task_bot, kaapi_tasksteal_arg_t );
        fprintf(file, ", stolen task:" );
        kaapi_task_print(file, arg->origin_task);
      }
      fputc('\n', file);
      ++count;
      ++task_bot;
      continue;
    }

    /* print the task */
    fprintf( file, "  [%04i]: ", count );
    kaapi_task_print(file, task_bot );

    ++count;
    ++task_bot;
  }

  fflush(file);
  return 0;
}
