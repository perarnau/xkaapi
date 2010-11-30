/*
** kaapi_fmt_predef.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
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

/**
*/
kaapi_format_id_t kaapi_format_taskregister_static( 
        kaapi_format_t*             fmt,
        kaapi_task_body_t           body,
        const char*                 name,
        size_t                      size,
        int                         count,
        const kaapi_access_mode_t   mode_param[],
        const kaapi_offset_t        offset_param[],
        const kaapi_format_t*       fmt_param[],
        const size_t                size_param[]
)
{
  kaapi_format_register( fmt, name );

  fmt->flag = KAAPI_FORMAT_STATIC_FIELD;
  
  fmt->size = (kaapi_uint32_t)size;
  fmt->_count_params    = count;
  
  fmt->_mode_params = malloc( sizeof(kaapi_access_mode_t)*count );
  kaapi_assert( fmt->_mode_params !=0);
  memcpy(fmt->_mode_params, mode_param, sizeof(kaapi_access_mode_t)*count );

  fmt->_off_params = malloc( sizeof(kaapi_offset_t)*count );
  kaapi_assert( fmt->_off_params !=0);
  memcpy(fmt->_off_params, offset_param, sizeof(kaapi_offset_t)*count );
  
  fmt->_fmt_params = malloc( sizeof(kaapi_format_t*)*count );
  kaapi_assert( fmt->_fmt_params !=0);
  memcpy(fmt->_fmt_params, fmt_param, sizeof(kaapi_format_t*)*count );

  fmt->_size_params = malloc( sizeof(size_t)*count );
  kaapi_assert( fmt->_size_params !=0);
  memcpy(fmt->_size_params, size_param, sizeof(size_t)*count );

  fmt->get_count_params=0;
  fmt->get_mode_param  =0;
  fmt->get_off_param   =0;
  fmt->get_fmt_param   =0;
  fmt->get_size_param  =0;
  
  memset(fmt->entrypoint, 0, sizeof(fmt->entrypoint));
  
  if (body !=0)
    kaapi_format_taskregister_body(fmt, body, KAAPI_PROC_TYPE_CPU);
  return fmt->fmtid;
}
