/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
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
#include "kaapic_impl.h"
#include <stdarg.h>
#include <string.h>


/* extern function, required by kaapif_spawn */
void kaapic_dfg_body(void* p, kaapi_thread_t* t)
{
  kaapic_task_info_t* const ti = (kaapic_task_info_t*)p;

#include "kaapic_dfg_switch.h"
  KAAPIC_DFG_SWITCH(ti);
}

static void* get_arg_wh
(const kaapic_task_info_t* ti, unsigned int i)
{
  const kaapic_arg_info_t* const ai = &ti->args[i];

  if (ai->mode != KAAPI_ACCESS_MODE_V)
  {
    const kaapi_data_t* const gd = (kaapi_data_t*)ai->access.data;
    return (void*)gd->ptr.ptr;
  }

  return ai->access.data;
}

static void kaapic_dfg_body_wh
(void* p, kaapi_thread_t* thread, kaapi_task_t* task)
{
  const kaapic_task_info_t* const ti = (const kaapic_task_info_t*)p;

#include "kaapic_dfg_wh_switch.h"
  KAAPIC_DFG_WH_SWITCH(ti);
}

/* format definition of C task */

static size_t kaapic_taskformat_get_count_params(
 const struct kaapi_format_t* f,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  return ti->nargs;
}

static kaapi_access_mode_t kaapic_taskformat_get_mode_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  const kaapi_access_mode_t m = ti->args[i].mode;
  return m;
}

static void* kaapic_taskformat_get_off_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  void* off = ti->args[i].access.data;
  return off;
}

static kaapi_access_t kaapic_taskformat_get_access_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  return ti->args[i].access;
}

static void kaapic_taskformat_set_access_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* p,
 const kaapi_access_t* a
)
{
  kaapic_task_info_t* const ti = p;
  ti->args[i].access = *a;
}

static const struct kaapi_format_t* kaapic_taskformat_get_fmt_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  const struct kaapi_format_t* const format = ti->args[i].format;
  return format;
}

static kaapi_memory_view_t kaapic_taskformat_get_view_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  return ti->args[i].view;
}

static void kaapic_taskformat_set_view_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* p,
 const kaapi_memory_view_t* v
)
{
  /* do nothing here if only 1-D view */
  kaapic_task_info_t* const ti = p;
  ti->args[i].view = *v;
}

__attribute__((unused)) 
static void kaapic_taskformat_reducor
(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* p,
 const void* q
)
{
  kaapi_abort();
}

__attribute__((unused)) 
static void kaapic_taskformat_redinit
(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* sp,
 void* p
)
{
  kaapi_abort();
}

__attribute__((unused))
static void kaapic_taskformat_get_task_binding(
 const struct kaapi_format_t* f,
 const kaapi_task_t* t,
 kaapi_task_binding_t* b
)
{
  b->type = KAAPI_BINDING_ANY;
}


void _kaapic_register_task_format(void)
{
  struct kaapi_format_t* format = kaapi_format_allocate();
  kaapi_format_taskregister_func
  (
    format,
    kaapic_dfg_body, 
    (kaapi_task_body_t)kaapic_dfg_body_wh,
    "kaapic_dfg_task",
    sizeof(kaapic_task_info_t),
    kaapic_taskformat_get_count_params,
    kaapic_taskformat_get_mode_param,
    kaapic_taskformat_get_off_param,
    kaapic_taskformat_get_access_param,
    kaapic_taskformat_set_access_param,
    kaapic_taskformat_get_fmt_param,
    kaapic_taskformat_get_view_param,
    kaapic_taskformat_set_view_param,
    0, /* reducor */
    0, /* redinit */
    0, /* task binding */
    0  /* get_splitter */
  );
}



/* dataflow interface */
int kaapic_spawn_ti( 
  kaapi_thread_t* thread, 
  const kaapic_spawn_attr_t*attr, 
  kaapi_task_body_t body, 
  kaapic_task_info_t* ti
)
{
  /* spawn the task */
  kaapi_task_init(kaapi_thread_toptask(thread), body, ti);
  if (attr == 0)
    kaapi_thread_pushtask(thread);
  else
    kaapi_thread_distribute_task(thread, attr->kid);
  return 0;
}

/* dataflow interface */
int kaapic_spawn(const kaapic_spawn_attr_t* attr, int32_t nargs, ...)
{
  kaapi_thread_t* thread = kaapi_self_thread();
  kaapic_task_info_t* ti;
  va_list va_args;
  size_t wordsize;
  unsigned int k;

  if (nargs > KAAPIC_MAX_ARGS) 
    return EINVAL;

  va_start(va_args, nargs);
    
  ti = kaapi_thread_pushdata_align(
    thread, sizeof(kaapic_task_info_t)+nargs*sizeof(kaapic_arg_info_t), sizeof(void*)
  );
  ti->body = va_arg(va_args, void (*)());
  ti->nargs = nargs;

  for (k = 0; k < nargs; ++k)
  {
    kaapic_arg_info_t* const ai = &ti->args[k];

    const uint32_t mode  = va_arg(va_args, int);
    void* const addr     = va_arg(va_args, void*);
    const uint32_t count = va_arg(va_args, int);
    const uint32_t type  = va_arg(va_args, int);

    switch (mode)
    {
      case KAAPIC_MODE_R:  
        ai->mode = KAAPI_ACCESS_MODE_R; 
        break;
      case KAAPIC_MODE_W:  
        ai->mode = KAAPI_ACCESS_MODE_W; 
        break;
      case KAAPIC_MODE_RW: 
        ai->mode = KAAPI_ACCESS_MODE_RW; 
        break;
      case KAAPIC_MODE_CW: 
        ai->mode = KAAPI_ACCESS_MODE_CW; 
        break;
      case KAAPIC_MODE_V:  
        if (count >1) 
          return EINVAL;
        ai->mode = KAAPI_ACCESS_MODE_V; 
        break;
      default: 
        return EINVAL;
    }

    switch (type)
    {
      case KAAPIC_TYPE_CHAR:
        wordsize = sizeof(int);
        ai->format = kaapi_char_format;
        break ;

      case KAAPIC_TYPE_INT:
        wordsize = sizeof(int);
        ai->format = kaapi_int_format;
        break ;

      case KAAPIC_TYPE_REAL:
        wordsize = sizeof(float);
        ai->format = kaapi_float_format;
        break ;

      case KAAPIC_TYPE_DOUBLE:
        wordsize = sizeof(double);
        ai->format = kaapi_double_format;
        break ;

      case KAAPIC_TYPE_PTR:
        wordsize = sizeof(void*);
        ai->format = kaapi_voidp_format;
        break ;

      default: 
        return EINVAL;
    }
    
    kaapi_access_init( &ai->access, addr );
    if (mode == KAAPIC_MODE_V)
    {
      /* can only pass exactly by value the size of a uintptr_t */
      kaapi_assert_debug( wordsize*count <= sizeof(uintptr_t) );
      memcpy(&ai->access.version, &addr, wordsize*count );  /* but count == 1 here */
    }

    ai->view = kaapi_memory_view_make1d(count, wordsize);
  }
  va_end(va_args);

  /* spawn the task */
  return kaapic_spawn_ti( thread, attr, kaapic_dfg_body, ti );
}
