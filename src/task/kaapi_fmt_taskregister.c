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


/**
*/
kaapi_format_id_t kaapi_format_taskregister( 
        kaapi_format_t*             fmt,
        kaapi_task_body_t           body,
        const char*                 name,
        size_t                      size,
        int                         count,
        const kaapi_access_mode_t   mode_param[],
        const kaapi_offset_t        offset_param[],
        const kaapi_format_t*       fmt_param[],
	      size_t (*get_param_size)(const kaapi_format_t*, unsigned int, const void*)
)
{
//  kaapi_format_t* fmt = (*fmt_fnc)();
  kaapi_format_register( fmt, name );

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

  fmt->get_param_size = get_param_size;
  memset(fmt->entrypoint, 0, sizeof(fmt->entrypoint));

  if (body !=0)
    kaapi_format_taskregister_body(fmt, body, KAAPI_PROC_TYPE_CPU);
  return fmt->fmtid;
}


/** TODO:
    - utilisation d'une autre structure de chainage que le format: 3 archi possible
    mais qu'un champ de link => seulement une archi dans la table de hash...
    - 
*/
kaapi_task_body_t kaapi_format_taskregister_body( 
        kaapi_format_t*             fmt,
        kaapi_task_body_t           body,
        int                         archi
)
{
  kaapi_uint8_t   entry;
  kaapi_format_t* head;

  if (body ==0) return fmt->entrypoint[archi];
  
  if (fmt->entrypoint[archi] ==body) return fmt->entrypoint[archi];
  fmt->entrypoint[archi] = body;
  if (archi == KAAPI_PROC_TYPE_DEFAULT)
    fmt->entrypoint[KAAPI_PROC_TYPE_DEFAULT] = fmt->default_body = body;

#if defined(KAAPI_DEBUG)
  fprintf(stdout, "[registerbody] Body:%p registered to name:%s\n", (void*)body, fmt->name );
  fflush(stdout);
#endif

  /* register it into hashmap: body -> fmt */
  entry = ((unsigned long)body) & 0xFF;
  head =  kaapi_all_format_bybody[entry];
  fmt->next_bybody = head;
  kaapi_all_format_bybody[entry] = fmt;

  /* already registered into hashmap: fmtid -> fmt */  
  return body;
}
