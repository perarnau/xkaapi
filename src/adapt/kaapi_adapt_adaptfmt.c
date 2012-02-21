/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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

/* Kaapi init adaptive may be suppress if the format objects can be initialized directly
   Note:
     - kaapi_taskadapt_body and kaapi_taskadapt_body share a kaapi_stealcontext_t data
     in RW mode. 
*/
static kaapi_format_t kaapi_format_task_adapt_body;

static inline kaapi_format_t* __resolve_format( const void* sp )
{
  const kaapi_taskadaptive_arg_t* arg = (const kaapi_taskadaptive_arg_t*)sp;
  if (arg->user_body ==0) return 0;
  return kaapi_format_resolvebybody(arg->user_body);
}

static inline void* __get_usersp( void* sp )
{
  kaapi_taskadaptive_arg_t* arg = (kaapi_taskadaptive_arg_t*)sp;
  return arg->user_sp;
}
static inline const void* __constget_usersp( const void* sp )
{
  kaapi_taskadaptive_arg_t* arg = (kaapi_taskadaptive_arg_t*)sp;
  return arg->user_sp;
}

static size_t 
_kaapi_adaptbody_get_count_params(const kaapi_format_t* fmt, const void* sp)
{ 
  const kaapi_format_t* user_fmt = __resolve_format(sp);
  return 1 
       + (user_fmt == 0 ? 0 : 
            kaapi_format_get_count_params( __resolve_format(sp), __constget_usersp(sp) ) ); 
}

static kaapi_access_mode_t 
_kaapi_adaptbody_get_mode_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  return ith == 0 ? KAAPI_ACCESS_MODE_RW 
                  : kaapi_format_get_mode_param(__resolve_format(sp), ith-1, __constget_usersp(sp));
}

static kaapi_access_t 
_kaapi_adaptbody_get_access_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_taskadaptive_arg_t* arg = (kaapi_taskadaptive_arg_t*)sp;
  return ith == 0 ? arg->shared_sc
                  : kaapi_format_get_access_param( __resolve_format(sp), ith-1, __constget_usersp(sp) );
}

static void 
_kaapi_adaptbody_set_access_param
  (const kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_access_t* a)
{
  if (ith ==0) 
  {
    kaapi_taskadaptive_arg_t* arg = (kaapi_taskadaptive_arg_t*)sp;
    arg->shared_sc = *a;
  }
  else 
    kaapi_format_set_access_param( __resolve_format(sp), ith-1, __get_usersp(sp), a );
}


static const kaapi_format_t*
_kaapi_adaptbody_get_fmt_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  return ith == 0 ? kaapi_char_format
                  : kaapi_format_get_fmt_param( __resolve_format(sp), ith-1, __constget_usersp(sp) );
}

static kaapi_memory_view_t 
_kaapi_adaptbody_get_view_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  return ith == 0 ? kaapi_memory_view_make1d( sizeof(kaapi_stealcontext_t), 1 )
                  : kaapi_format_get_view_param( __resolve_format(sp), ith-1, __constget_usersp(sp) );
}

static void 
_kaapi_adaptbody_set_view_param(
    const kaapi_format_t* fmt, 
    unsigned int ith, 
    void* sp, 
    const kaapi_memory_view_t* view
)
{
  if (ith >0)
    kaapi_format_set_view_param( __resolve_format(sp), ith-1, __get_usersp(sp), view );
  /* else nothing to do: never reallocated because, 1D view ! */
}

/* good name ! 
   Wrapper between format function and the user splitter
*/
static int kaapi_adaptivetask_wrapper_splitter( 
    kaapi_task_t*                 pc, /* this is the adaptive task on which splitter is called */
    void*                         sp,
    kaapi_listrequest_t*          lrequests, 
    kaapi_listrequest_iterator_t* lrrange
)
{
  /* - pc is kaapi_taskadapt_body 
  */
  kaapi_taskadaptive_arg_t* arg = (kaapi_taskadaptive_arg_t*)sp;
  kaapi_adaptivetask_splitter_t splitter = arg->user_splitter;
  kaapi_assert_debug( pc !=0 );

  /* call the splitter */
  if (splitter !=0)
    return splitter( 
        pc,
        arg->user_sp,
        lrequests,
        lrrange
    );
  return EINVAL;
}

static kaapi_adaptivetask_splitter_t 
_kaapi_adaptbody_get_splitter(const struct kaapi_format_t* fmt, const void* sp)
{ 
  return kaapi_adaptivetask_wrapper_splitter; 
  //const kaapi_taskadaptive_arg_t* arg = (const kaapi_taskadaptive_arg_t*)sp;
  //return arg->user_splitter;
}

void kaapi_init_adapfmt(void)
{
  kaapi_format_taskregister_func( 
    &kaapi_format_task_adapt_body, 
    (kaapi_task_body_t)kaapi_taskadapt_body,
    0,
    "kaapi_taskadapt_body",
    sizeof(kaapi_taskadaptive_arg_t),
    _kaapi_adaptbody_get_count_params,
    _kaapi_adaptbody_get_mode_param,
    0  /* (*get_off_param) */,
    _kaapi_adaptbody_get_access_param,
    _kaapi_adaptbody_set_access_param,
    _kaapi_adaptbody_get_fmt_param,
    _kaapi_adaptbody_get_view_param,
    _kaapi_adaptbody_set_view_param,
    0  /* (*reducor) */,
    0  /* (*redinit) */,
    0  /* (*get_task_binding) */,
    _kaapi_adaptbody_get_splitter
  );
}
