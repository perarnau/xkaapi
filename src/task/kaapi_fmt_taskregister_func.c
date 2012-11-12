/*
** kaapi_fmt_predef.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include <string.h>

static void kaapi_format_default_get_task_binding 
  (const struct kaapi_format_t* f, const void* task, kaapi_task_binding_t* b)
{ b->type = KAAPI_BINDING_ANY; }

/**
*/
kaapi_format_id_t kaapi_format_taskregister_func( 
    struct kaapi_format_t*        fmt, 
    kaapi_task_body_t             body,
    kaapi_task_body_t             bodywh,
    const char*                   name,
    size_t                      (*get_size)(const struct kaapi_format_t*, const void*),
    void                        (*task_copy)(const struct kaapi_format_t*, void*, const void*),
    size_t                      (*get_count_params)(const struct kaapi_format_t*, const void*),
    kaapi_access_mode_t         (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*),
    void*                       (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*),
    kaapi_access_t              (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*),
    void                        (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*),
    const struct kaapi_format_t*(*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*),
    kaapi_memory_view_t         (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*),
    void                        (*set_view_param)  (const struct kaapi_format_t*, unsigned int, void*, const kaapi_memory_view_t*),
    void                        (*reducor )        (const struct kaapi_format_t*, unsigned int, void*, const void*),
    void                        (*redinit )        (const struct kaapi_format_t*, unsigned int, const void*, void* ),
    void			            (*get_task_binding)(const struct kaapi_format_t*, const void*, kaapi_task_binding_t*),
  kaapi_adaptivetask_splitter_t	(*get_splitter)(const struct kaapi_format_t*, const void* )
)
{
//  kaapi_format_t* fmt = (*fmt_fnc)();
  kaapi_format_register( fmt, name );

  fmt->flag = KAAPI_FORMAT_DYNAMIC_FIELD;
  
  fmt->get_size         = get_size;
  fmt->task_copy        = task_copy;
  fmt->get_count_params = get_count_params;
  fmt->get_mode_param   = get_mode_param;
  fmt->get_off_param    = get_off_param;
  fmt->get_access_param = get_access_param;
  fmt->set_access_param = set_access_param;
  fmt->get_fmt_param    = get_fmt_param;
  fmt->get_view_param   = get_view_param;
  fmt->set_view_param   = set_view_param;
  fmt->reducor          = reducor;
  fmt->redinit          = redinit;
  fmt->get_task_binding	= get_task_binding;
  if (get_task_binding ==0)
    fmt->get_task_binding = kaapi_format_default_get_task_binding;
  fmt->get_splitter	    = get_splitter;
  
  memset(fmt->entrypoint, 0, sizeof(fmt->entrypoint));
  
  if (body !=0)
    kaapi_format_taskregister_body(fmt, body, bodywh, KAAPI_PROC_TYPE_CPU);
  return fmt->fmtid;
}
