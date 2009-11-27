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
static const char* getflagsname( unsigned int flag )
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



/** 
*/
int kaapi_stack_print  ( int fd, kaapi_stack_t* stack )
{
  FILE* file;
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  const kaapi_format_t* fmt;
  int count;
  int i;
  char isexec;

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
    if (task_bot->body ==0) fmt = kaapi_format_resolvebybody( (kaapi_task_body_t)task_bot->format );
    else if (task_bot->body == &kaapi_suspend_body) fmt = task_bot->format;
    else fmt = kaapi_format_resolvebybody( task_bot->body );
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

      if (task_bot->body ==0) isexec = 'X';
      else if (task_bot->body == &kaapi_suspend_body) isexec = 'S';
      else if (task_bot->body == &kaapi_aftersteal_body) isexec = 'M';
      else isexec = 'I';
      fprintf( file, "  [%i]: @%p |%c%s|, name:%s, splitter:%p", 
            count, 
            (void*)task_bot,
            isexec, 
            getflagsname(task_bot->flag), 
            fname, 
            (void*)task_bot->splitter );
      
      if (task_bot->body == &kaapi_retn_body)
      {
        kaapi_frame_t* frame = kaapi_task_getargst( task_bot, kaapi_frame_t );
        fprintf(file, ", pc:%p, sp:%p, spd:%p", frame->pc, frame->sp, frame->sp_data );
      }
      fputc('\n', file);
      ++count;
      ++task_bot;
      continue;
    }
    
    if (task_bot->body ==0) isexec = 'X';
    else if (task_bot->body == &kaapi_suspend_body) isexec = 'S';
    else if (task_bot->body == &kaapi_aftersteal_body) isexec = 'M';
    else isexec = 'I';
    fprintf( file, "  [%i]: @%p |%c%s|, name:%s, splitter:%p, #p:%i, mode: ", 
          count, 
          (void*)task_bot, 
          isexec,
          getflagsname(task_bot->flag), 
          fmt->name, 
          (void*)task_bot->splitter,
          fmt->count_params );
          
    /* access mode */
    for (i=0; i<fmt->count_params; ++i)
    {
      const kaapi_format_t* fmt_param = fmt->fmt_params[i];
      kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(fmt->mode_params[i]);   
      switch (m) {
        case KAAPI_ACCESS_MODE_V:
          fputc('v', file);
        break;
        case KAAPI_ACCESS_MODE_R:
          fputc('r', file);
        break;
        case KAAPI_ACCESS_MODE_W:
          fputc('w', file);
        break;
        case KAAPI_ACCESS_MODE_CW:
          fputc('c', file);
        break;
        case KAAPI_ACCESS_MODE_RW:
          fputc('x', file);
        break;
        default:
          fputc('!', file );
      }
      if (m == KAAPI_ACCESS_MODE_V)
      {
        void* data = (void*)(fmt->off_params[i] + (char*)task_bot->sp);
        fprintf(file, "<%s>, @:%p", fmt_param->name, data );
      }
      else 
      {
        kaapi_access_t* access = (kaapi_access_t*)(fmt->off_params[i] + (char*)task_bot->sp);
        kaapi_gd_t* gd = ((kaapi_gd_t*)access->data)-1;
        fprintf(file, "<%s>, @:%p, a:%p", fmt_param->name, (void*)gd, access->data );
      }
      if (i <fmt->count_params-1)
      {
        fputs("|| ", file );
      }
      else
        fputc('\n', file );
    }

    ++count,
    ++task_bot;
  }

  return 0;
}
