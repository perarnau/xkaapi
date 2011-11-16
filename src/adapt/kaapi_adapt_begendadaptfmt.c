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
static kaapi_format_t kaapi_format_task_begendadapt_body;


static size_t 
_kaapi_begendadaptbody_get_count_params(const kaapi_format_t* fmt, const void* sp)
{ return 1 ; }

static kaapi_access_mode_t 
_kaapi_begendadaptbody_get_mode_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( ith == 0 );
  return KAAPI_ACCESS_MODE_RW;
}

static kaapi_access_t 
_kaapi_begendadaptbody_get_access_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_taskmerge_arg_t* arg = (kaapi_taskmerge_arg_t*)sp;
  kaapi_assert_debug( ith == 0 );
  return arg->shared_sc;
}

static void 
_kaapi_begendadaptbody_set_access_param
  (const kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_access_t* a)
{
  kaapi_taskmerge_arg_t* arg = (kaapi_taskmerge_arg_t*)sp;
  kaapi_assert_debug( ith == 0 );
  arg->shared_sc = *a;
}


static const kaapi_format_t*
_kaapi_begendadaptbody_get_fmt_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( ith == 0 );
  return kaapi_char_format;
}

static kaapi_memory_view_t 
_kaapi_begendadaptbody_get_view_param(const kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( ith == 0 );
  return kaapi_memory_view_make1d( sizeof(kaapi_stealcontext_t), 1 );
}

static void 
_kaapi_begendadaptbody_set_view_param
  (const kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_memory_view_t* view)
{
  kaapi_assert_debug( ith == 0 );
  /* else nothing to do: never reallocated because, 1D view ! */
}

void kaapi_init_begendadapfmt(void)
{
  kaapi_format_taskregister_func( 
    &kaapi_format_task_begendadapt_body, 
    (kaapi_task_body_t)kaapi_taskbegendadapt_body,
    0,
    "kaapi_taskbegendadapt_body",
    sizeof(kaapi_taskbegendadaptive_arg_t),
    _kaapi_begendadaptbody_get_count_params,
    _kaapi_begendadaptbody_get_mode_param,
    0  /* (*get_off_param) */,
    _kaapi_begendadaptbody_get_access_param,
    _kaapi_begendadaptbody_set_access_param,
    _kaapi_begendadaptbody_get_fmt_param,
    _kaapi_begendadaptbody_get_view_param,
    _kaapi_begendadaptbody_set_view_param,
    0  /* (*reducor) */,
    0  /* (*redinit) */,
    0  /* (*get_task_binding) */,
    0  /* (*get_splitter) */
  );
}
