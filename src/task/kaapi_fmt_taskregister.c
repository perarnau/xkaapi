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
kaapi_format_t* kaapi_all_format_bybody[256] = 
{
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
static kaapi_task_bodyid_t bodyidcounter = KAAPI_TASK_BODY_USER_BASE;
/**
*/
kaapi_format_id_t kaapi_format_taskregister( 
        kaapi_format_t*           (*fmt_fnc)(void),
        kaapi_task_bodyid_t         bodyid,
        kaapi_task_body_t           body,
        const char*                 name,
        size_t                      size,
        int                         count,
        const kaapi_access_mode_t   mode_param[],
        const kaapi_offset_t        offset_param[],
        const kaapi_format_t*       fmt_param[]
)
{
  kaapi_format_t* fmt = (*fmt_fnc)();
  kaapi_format_register( fmt, name );

  if (bodyid == -1) 
    bodyid = bodyidcounter++;
  kaapi_assert( (KAAPI_TASK_BODY_USER_BASE<= bodyid) && (bodyid <= 0xFF));
  kaapi_bodies[bodyid] = body;
  fmt->bodyid = bodyid;
  fmt->entrypoint[KAAPI_PROC_TYPE_CPU] = body;
  fmt->count_params    = count;
  
  fmt->mode_params = malloc( sizeof(kaapi_access_mode_t)*count );
  kaapi_assert(  fmt->mode_params !=0);
  memcpy(fmt->mode_params, mode_param, sizeof(kaapi_access_mode_t)*count );

  fmt->off_params = malloc( sizeof(kaapi_offset_t)*count );
  kaapi_assert(  fmt->off_params !=0);
  memcpy(fmt->off_params, offset_param, sizeof(kaapi_offset_t)*count );
  
  fmt->fmt_params = malloc( sizeof(kaapi_format_t*)*count );
  kaapi_assert(  fmt->fmt_params !=0);
  memcpy(fmt->fmt_params, fmt_param, sizeof(kaapi_format_t*)*count );

  fmt->size = size;

  /* register it into hashmap: body -> fmt */
  kaapi_all_format_bybody[bodyid] = fmt;
  
  /* already registered into hashmap: fmtid -> fmt */  
  return fmt->fmtid;
}
