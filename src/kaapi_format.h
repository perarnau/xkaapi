/*
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
#ifndef _KAAPI_FORMAT_H_
#define _KAAPI_FORMAT_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif

#include "config.h"
#include "kaapi_error.h"
#include "kaapi_defs.h"
#include "kaapi.h"


struct kaapi_taskdescr_t;
struct kaapi_format_t;

/** Global hash table of all formats: body -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_bybody[256];

/** Global hash table of all formats: fmtid -> fmt
*/
extern struct kaapi_format_t* kaapi_all_format_byfmtid[256];

/* ============================= Format for task/data structure ============================ */

typedef enum kaapi_format_flag_t {
  KAAPI_FORMAT_STATIC_FIELD,   /* the format of the task is interpreted using static offset/fmt etc */ 
  KAAPI_FORMAT_DYNAMIC_FIELD      /* the format is interpreted using the function, required for variable args tasks */
} kaapi_format_flag_t;

/** \ingroup TASK
    Kaapi task format
    The format should be 1/ declared 2/ register before any use in task.
    The format object is only used in order to interpret stack of task.    
*/
typedef struct kaapi_format_t {
  kaapi_format_id_t          fmtid;                                   /* identifier of the format */
  short                      isinit;                                  /* ==1 iff initialize */
  const char*                name;                                    /* debug information */
  const char*                name_dot;                                /* name for DOT */
  const char*                color_dot;                               /* color for DOT */
  
  /* flag to indicate how to interpret the following fields */
  kaapi_format_flag_t        flag;
  
  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_STATIC_FIELD */
  uint32_t                   size;                                    /* sizeof the object */  
  void                       (*cstor)( void* dest);
  void                       (*dstor)( void* dest);
  void                       (*cstorcopy)( void* dest, const void* src);
  void                       (*copy)( void* dest, const void* src);
  void                       (*assign)( void* dest, const kaapi_memory_view_t* view_dest, const void* src, const kaapi_memory_view_t* view_src);
  void                       (*print)( FILE* file, const void* src);

  /* only if it is a format of a task  */
  kaapi_task_body_t          default_body;                            /* iff a task used on current node */
  kaapi_task_body_t          entrypoint[KAAPI_PROC_TYPE_MAX];         /* maximum architecture considered in the configuration */
  kaapi_task_body_t          entrypoint_wh[KAAPI_PROC_TYPE_MAX];      /* same as entrypoint, except that shared params are handle to memory location */
  kaapi_task_body_t          alpha_body;				/* alpha function of acceleration */

  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_STATIC_FIELD */
  int                         _count_params;                          /* number of parameters */
  kaapi_access_mode_t        *_mode_params;                           /* only consider value with mask 0xF0 */
  kaapi_offset_t             *_off_params;                            /*access to the i-th parameter: a value or a shared */
  kaapi_offset_t             *_off_versions;                          /*access to the i-th parameter: a value or a shared */
  struct kaapi_format_t*     *_fmt_params;                            /* format for each params */
  kaapi_memory_view_t        *_view_params;                           /* sizeof of each params */
  kaapi_reducor_t            *_reducor_params;                        /* array of reducor in case of cw */
  kaapi_redinit_t            *_redinit_params;                        /* array of redinit in case of cw */
  kaapi_task_binding_t       _task_binding;

  /* case of format for a structure or for a task with flag= KAAPI_FORMAT_FUNC_FIELD
     - the unsigned int argument is the index of the parameter 
     - the last argument is the pointer to the sp data of the task
  */
  size_t                (*get_size)(const struct kaapi_format_t*, const void*);
  void                  (*task_copy)(const struct kaapi_format_t*, void*, const void*);
  size_t                (*get_count_params)(const struct kaapi_format_t*, const void*);
  kaapi_access_mode_t   (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*);
  void*                 (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*);
  kaapi_access_t        (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*);
  void                  (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*);
  const struct kaapi_format_t*(*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*);
  kaapi_memory_view_t   (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*);
  void (*set_view_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_memory_view_t* );

  void                  (*reducor )        (const struct kaapi_format_t*, unsigned int, void* sp, const void* value);
  void                  (*redinit )        (const struct kaapi_format_t*, unsigned int, const void* sp, void* value );
  void			            (*get_task_binding)(const struct kaapi_format_t*, const void* sp, kaapi_task_binding_t*);
  kaapi_adaptivetask_splitter_t	(*get_splitter)(const struct kaapi_format_t*, const void* sp);

  /* fields to link the format is the internal tables */
  struct kaapi_format_t      *next_bybody;                            /* link in hash table */
  struct kaapi_format_t      *next_byfmtid;                           /* link in hash table */
  
  /* only for Monotonic bound format */
  int    (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                      const void* value, const struct kaapi_format_t* fmtvalue );
} kaapi_format_t;



/* Helper to interpret the format 
*/
static inline 
size_t                kaapi_format_get_size(const struct kaapi_format_t* fmt, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->size;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_size)(fmt, sp);
}

static inline 
void                kaapi_format_task_copy(const struct kaapi_format_t* fmt, void* sp_dest, const void* sp_src)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    memcpy(sp_dest, sp_src,  fmt->size);
  }
  else {
    kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
    (*fmt->task_copy)(fmt, sp_dest, sp_src);
  }
}

static inline 
size_t                kaapi_format_get_count_params(const struct kaapi_format_t* fmt, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_count_params;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_count_params)(fmt, sp);
}

static inline 
kaapi_access_mode_t   kaapi_format_get_mode_param (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_mode_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_mode_param)(fmt, ith, sp);
}

static inline 
void*                 kaapi_format_get_data_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) == KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_off_params[ith] + (char*)sp;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_off_param)(fmt, ith, sp);
}

static inline 
kaapi_access_t         kaapi_format_get_access_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) != KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) {
    kaapi_access_t retval;
    retval.data    = *(void**)(fmt->_off_params[ith] + (char*)sp);
    retval.version = *(void**)(fmt->_off_versions[ith] + (char*)sp);
    return retval;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_access_param)(fmt, ith, sp);
}

static inline 
void         kaapi_format_set_access_param  (const struct kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_access_t* a)
{
  kaapi_assert_debug( KAAPI_ACCESS_GET_MODE(kaapi_format_get_mode_param(fmt, ith, sp)) != KAAPI_ACCESS_MODE_V );
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    *(void**)(fmt->_off_params[ith] + (char*)sp) = a->data;
    *(void**)(fmt->_off_versions[ith] + (char*)sp) = a->version;
    return;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->set_access_param)(fmt, ith, sp, a);
}


static inline 
const struct kaapi_format_t* kaapi_format_get_fmt_param  (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_fmt_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_fmt_param)(fmt, ith, sp);
}

static inline 
kaapi_memory_view_t kaapi_format_get_view_param (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return fmt->_view_params[ith];
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_view_param)(fmt, ith, sp);
}

static inline 
void kaapi_format_set_view_param (const struct kaapi_format_t* fmt, unsigned int ith, void* sp, const kaapi_memory_view_t* view)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    kaapi_assert(0);
    return;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->set_view_param)(fmt, ith, sp, view);
}

static inline 
void          kaapi_format_reduce_param (const struct kaapi_format_t* fmt, unsigned int ith, void* sp, const void* value)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    (*fmt->_reducor_params[ith])( *(void**)(fmt->_off_params[ith] + (char*)sp), value);
    return;
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->reducor)(fmt, ith, sp, value);
}

static inline 
void kaapi_format_redinit_neutral (const struct kaapi_format_t* fmt, unsigned int ith, const void* sp, void* value )
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
  {
    (*fmt->_redinit_params[ith])( value );
  }
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  (*fmt->redinit)(fmt, ith, sp, value);
}


static inline void kaapi_format_get_task_binding 
  (const struct kaapi_format_t* fmt, const void* sp, kaapi_task_binding_t* b)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) 
    *b = fmt->_task_binding;
  else {
    if (fmt->get_task_binding == 0)
      b->type = KAAPI_BINDING_ANY; 
    fmt->get_task_binding(fmt, sp, b );
  }
}


static inline 
kaapi_adaptivetask_splitter_t kaapi_format_get_splitter(const struct kaapi_format_t* fmt, const void* sp)
{
  if (fmt->flag == KAAPI_FORMAT_STATIC_FIELD) return 0;
  kaapi_assert_debug( fmt->flag == KAAPI_FORMAT_DYNAMIC_FIELD );
  return (*fmt->get_splitter)(fmt, sp);
}

static inline 
kaapi_task_body_t kaapi_format_get_task_bodywh_by_arch
(
  const kaapi_format_t*	const	fmt, 
  unsigned int arch
)
{
  return fmt->entrypoint_wh[arch];
}

extern struct kaapi_format_t* kaapi_staticschedtask_format;

static inline 
kaapi_task_body_t kaapi_format_get_task_body_by_arch
(
  const kaapi_format_t*	const	fmt, 
  unsigned int arch
)
{
  return fmt->entrypoint[arch];
}

static inline int
kaapi_format_is_staticschedtask( const kaapi_format_t*	const	fmt )
{
  return ( fmt == kaapi_staticschedtask_format);
}

/** Initialise default formats
*/
extern void kaapi_init_basicformat(void);


#define KAAPI_DECLEXTERN_BASICTYPEFORMAT( formatobject ) \
  extern kaapi_format_t formatobject##_object;

KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_char_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_short_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_int_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_long_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_longlong_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_uchar_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_ushort_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_uint_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_ulong_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_ulonglong_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_float_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_double_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_longdouble_format)
KAAPI_DECLEXTERN_BASICTYPEFORMAT(kaapi_voidp_format)


#if defined(__cplusplus)
}
#endif

#endif
