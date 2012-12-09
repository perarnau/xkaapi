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
#include <limits.h>
#include <float.h>
#include <ffi.h>
/* We need some constant from the fortran interface */
#include "../kaapif/kaapif_inc.h"

/* Define all basic types. Provided information will be extracted by
 * futher macros, but all knowledge about these types are put here.
 *
 * This macro can be used as the following:
 * #define _KAAPI_m(a,b,c,...)  body macro using a, b and c
 * _KAAPIC_TYPE_ITER(_KAAPI_m, ;)
 * #undef _KAAPI_m(a,b,c,...)
 *
 * Note: it is better to define futher macros (such as _KAAPI_m before)
 * with an ellips (...) at the end, so that additionnal features can
 * easily be added here with having to update the parameter list of
 * all macros.
 *
 * For each type, the following parameters are available:
 * TYPE: uniq uppercase type identifier
 * type: uniq lower case type identifier
 * ctype: the targetted C type
 * promtype: the C type into which the 'ctype' is promoted when used
 *   in variadic function
 * redop: 0: no reduction available
 *        1: integer reductions available
 *        2: float reduction available
 * ffi_type: suffix of corresponding ffi_type_*
 *   note: ffi_type_char does not exist. Failback to schar for now.
 *         ffi_type_[us]longlong do not exist. Failback to [us]int64 for now.
 */

#define _KAAPIC_TYPE_ITER_INTEGERS(m, sep)				\
  m(CHAR, char, char, int,						\
    1, schar) sep							\
  m(SCHAR, schar, signed char, int,					\
    1, schar) sep							\
  m(SHRT, shrt, short, int,						\
    1, sshort) sep							\
  m(INT, int, int, int,							\
    1, sint) sep							\
  m(LONG, long, long, long,						\
    1, slong) sep							\
  m(LLONG, llong, long long, long long,					\
    1, sint64) sep							\
  m(INT8, int8, int8_t, int,						\
    1, sint8) sep							\
  m(INT16, int16, int16_t, int,						\
    1, sint16) sep							\
  m(INT32, int32, int32_t, int32_t,					\
    1, sint32) sep							\
  m(INT64, int64, int64_t, int64_t,					\
    1, sint64) sep							\
  m(UCHAR, uchar, unsigned char, unsigned,				\
    1, uchar) sep							\
  m(USHRT, ushrt, unsigned short, unsigned,				\
    1, ushort) sep							\
  m(UINT, uint, unsigned int, unsigned int,				\
    1, uint) sep							\
  m(ULONG, ulong, unsigned long, unsigned long,				\
    1, ulong) sep							\
  m(ULLONG, ullong, unsigned long long, unsigned long long,		\
    1, uint64) sep							\
  m(UINT8, uint8, uint8_t, unsigned,					\
    1, uint8) sep							\
  m(UINT16, uint16, uint16_t, unsigned,					\
    1, uint16) sep							\
  m(UINT32, uint32, uint32_t, uint32_t,					\
    1, uint32) sep							\
  m(UINT64, uint64, uint64_t, uint64_t,					\
    1, uint64)

#define _KAAPIC_TYPE_ITER_FLOATS(m, sep)				\
  m(FLT, flt, float, double,						\
    2, float) sep							\
  m(DBL, dbl, double, double,						\
    2, double) sep							\
  m(LDBL, ldbl, long double, long double,				\
    2, longdouble)

#define _KAAPIC_TYPE_ITER_OTHERS(m, sep)				\
  m(PTR, voidp, void*, void*,						\
    0, pointer)

/* Defining the min macro for unsigned type so that code can be
 * generated with the same macro for signed and unsigned types
 */
#define UCHAR_MIN 0
#define USHRT_MIN 0
#define UINT_MIN 0
#define ULONG_MIN 0
#define ULLONG_MIN 0
#define UINT8_MIN 0
#define UINT16_MIN 0
#define UINT32_MIN 0
#define UINT64_MIN 0

#define _KAAPIC_TYPE_ITER(m, sep)					\
  _KAAPIC_TYPE_ITER_INTEGERS(m, sep) sep				\
  _KAAPIC_TYPE_ITER_FLOATS(m, sep) sep					\
				   _KAAPIC_TYPE_ITER_OTHERS(m, sep)

#define _KAAPIC_COMMA ,

#define _KAAPIC_TYPE_ITER_INTEGERS_COMMA(m)	\
  _KAAPIC_TYPE_ITER_INTEGERS(m, _KAAPIC_COMMA)

#define _KAAPIC_TYPE_ITER_FLOATS_COMMA(m)	\
    _KAAPIC_TYPE_ITER_FLOATS(m, _KAAPIC_COMMA)

#define _KAAPIC_TYPE_ITER_OTHERS_COMMA(m)	\
    _KAAPIC_TYPE_ITER_OTHERS(m, _KAAPIC_COMMA)

#define _KAAPIC_TYPE_ITER_COMMA(m)					\
  _KAAPIC_TYPE_ITER_INTEGERS(m, _KAAPIC_COMMA) ,			\
    _KAAPIC_TYPE_ITER_FLOATS(m, _KAAPIC_COMMA) ,			\
    _KAAPIC_TYPE_ITER_OTHERS(m, _KAAPIC_COMMA)

/*
*/
typedef struct kaapic_arg_sig
{
  struct {
    kaapi_access_mode_t        mode : 16;
    int                        type : 8;
    int                        redop: 8;
  } /* anonymous */;
  kaapi_memory_view_t          view;
  const struct kaapi_format_t* format;
  /* In case of a V argument, store the offset from
   * the 'values' area where to store the value.
   *
   * The voffset value is set to -1 if the value is
   * stored into the 'version' field of the access structure.
   */
  ptrdiff_t                    voffset;
} kaapic_arg_sig_t;

/* FFI must be used to start the body task */
#define _KAAPIC_TS_USE_FFI (1<<0)
/* Task has scratch parameters, they must be proceeded at body start */
#define _KAAPIC_TS_HAS_SCRATCH (1<<1)
/* This signature includes the view */
#define _KAAPIC_TS_HAS_COUNT (1<<2)
/* The body is a fortran function, we must respect Fortran ABI */
#define _KAAPIC_TS_FORTRAN_ABI (1<<3)

#define KAAPIC_TS(ts, name)			\
  ((ts)->flags & _KAAPIC_TS_##name)
#define _KAAPIC_USE_FFI(ts) KAAPIC_TS(ts, USE_FFI)

typedef struct kaapic_task_sig
{
  uint16_t           nargs;
  uint16_t           flags;
  uint16_t           values_align;
  size_t             values_size;
  ffi_cif            cif;
  kaapic_arg_sig_t   args[];
} kaapic_task_sig_t;

typedef struct kaapic_arg_info_t
{
  kaapi_memory_view_t          view;

  /* kaapi versionning for shared pointer also used to store
     address of  the value for by-value argument it its size is
     small enought
  */
  kaapi_access_t              access;

} kaapic_arg_info_t;

/*
*/
typedef struct kaapic_task_info
{
  void             (*body)();
  kaapic_task_sig_t *sig;
  kaapic_arg_info_t  args[];
} kaapic_task_info_t;


static const kaapi_access_mode_t modec2modek[] =
{
  [KAAPIC_MODE_R] = KAAPI_ACCESS_MODE_R,
  [KAAPIC_MODE_W] = KAAPI_ACCESS_MODE_W,
  [KAAPIC_MODE_RW]= KAAPI_ACCESS_MODE_RW,
  [KAAPIC_MODE_CW]= KAAPI_ACCESS_MODE_CW,
  [KAAPIC_MODE_V] = KAAPI_ACCESS_MODE_V,
  [KAAPIC_MODE_T] = KAAPI_ACCESS_MODE_SCRATCH,
  [KAAPIC_MODE_S] = KAAPI_ACCESS_MODE_STACK
};

static const enum kaapic_mode modef2modec[KAAPIF_MODE_MAX] =
{
  [KAAPIF_MODE_R] = KAAPIC_MODE_R,
  [KAAPIF_MODE_W] = KAAPIC_MODE_W,
  [KAAPIF_MODE_RW]= KAAPIC_MODE_RW,
  [KAAPIF_MODE_V] = KAAPIC_MODE_V,
};

static const enum kaapic_mode typef2typec[KAAPIF_TYPE_MAX] =
{
  [KAAPIF_TYPE_CHAR] = KAAPIC_TYPE_CHAR,
  [KAAPIF_TYPE_INT] = KAAPIC_TYPE_INT,
  [KAAPIF_TYPE_REAL] = KAAPIC_TYPE_FLT,
  [KAAPIF_TYPE_DOUBLE] = KAAPIC_TYPE_DBL,
  [KAAPIF_TYPE_PTR] = KAAPIC_TYPE_PTR,
};


/* Features optimized at compile time when inlining */
#define _KAAPIC_DFG_BODY_WH (1<<0)
#define _KAAPIC_DFG_BODY_SCRATCH (1<<1)
#define _KAAPIC_DFG_BODY_FORTRAN_ABI (1<<2)

#define _FEATURE(name) \
  (inline_mode & _KAAPIC_DFG_BODY_##name)

/* generic get_arg function
 * This function will always be inlined
 * inline_mode will then be known at compiled time and optimized
 * This allows to write only one function (easy to maintain, avoid to
 * forget to replicate a bug fix, ...) to cover all cases with no
 * impact on the performance of the real functions
 */
static inline void* get_arg(
  int inline_mode,
  kaapic_task_info_t* ti,
  const kaapic_task_sig_t* const ts,
  unsigned int i) __attribute__((always_inline));
static inline void* get_arg(
  int inline_mode,
  kaapic_task_info_t* ti,
  const kaapic_task_sig_t* const ts,
  unsigned int i)
{
  kaapic_arg_info_t* const ai = &ti->args[i];
  const kaapic_arg_sig_t* const as = &ts->args[i];
  void* pdata;

  if (as->mode == KAAPI_ACCESS_MODE_V) {
    if (_FEATURE(FORTRAN_ABI)) {
      /* In fortran, values are passed by reference */
      pdata = &ai->access.data;
    } else {
      pdata = ai->access.data;
    }
  } else {
    if (_FEATURE(WH)) {
      const kaapi_data_t* const gd = (kaapi_data_t*)ai->access.data;
      pdata=(void*)&gd->ptr.ptr;
    } else {
      pdata=&ai->access.data;
    }
  }
  return pdata;
}

/* generic kaapic_dfg_body function
 * This function will always be inlined
 * inline_mode will then be known at compiled time and optimized
 * This allows to write only one function (easy to maintain, avoid to
 * forget to replicate a bug fix, ...) to cover all cases with no
 * impact on the performance of the real functions
 */
static inline void __kaapic_dfg_body(
  int inline_mode,
  void* p) __attribute__((always_inline));
static inline void __kaapic_dfg_body(
  int inline_mode,
  void* p)
{
  kaapic_task_info_t* const ti = (kaapic_task_info_t*)p;
  kaapic_task_sig_t* const ts = ti->sig;
  const unsigned int nargs = ts->nargs;

  if (KAAPIC_TS(ts, HAS_SCRATCH)) {
    /* process scratch mode */
    int scratch_count        = 0;
    kaapi_processor_t* kproc = 0;
    for (unsigned int i=0; i<nargs; ++i)
    {
      if (ts->args[i].mode & KAAPI_ACCESS_MODE_T)
      {
	if (kproc == 0) kproc = kaapi_get_current_processor();
	size_t szarg = kaapi_memory_view_size(&ts->args[i].view);
	// TODO: PB with get_arg_wh ?
	ti->args[i].access.data = _kaapi_gettemporary_data(kproc, scratch_count, szarg);
	++scratch_count;
      }
    }
  }

  if (KAAPIC_TS(ts, USE_FFI)) {
    unsigned int nargs = ts->nargs;
    void* pvalues[nargs];
    for (unsigned int i=0; i<nargs; ++i) {
      pvalues[i]= get_arg(inline_mode,ti, ts, i) ;
    }
    ffi_call(&ts->cif, ti->body, NULL, pvalues);
    return;
  }

#include "kaapic_dfg_switch.h"
  KAAPIC_DFG_SWITCH(*(void**)get_arg,inline_mode,ti,ts);
}
#undef _FEATURE

static void kaapic_dfg_body(void* p, kaapi_thread_t* t)
{
  return __kaapic_dfg_body(0, p);
}

static void kaapic_dfg_body_wh
(void* p, kaapi_thread_t* thread, kaapi_task_t* task)
{
  return __kaapic_dfg_body(_KAAPIC_DFG_BODY_WH, p);
}

/* same as kaapic_dfg_body but use fortran ABI */
static void kaapic_dfg_body_fortran(void* p, kaapi_thread_t* t)
{
  return __kaapic_dfg_body(_KAAPIC_DFG_BODY_FORTRAN_ABI, p);
}

/* same as kaapic_dfg_body_wh but use fortran ABI */
static void kaapic_dfg_body_wh_fortran(void* p, kaapi_thread_t* t, kaapi_task_t* task)
{
  return __kaapic_dfg_body(_KAAPIC_DFG_BODY_WH | _KAAPIC_DFG_BODY_FORTRAN_ABI, p);
}

/* format definition of C task */
static size_t kaapic_taskformat_get_size(const struct kaapi_format_t* fmt, const void* sp)
{
  const kaapic_task_info_t* const ti = (const kaapic_task_info_t*)sp;
  const kaapic_task_sig_t* const ts = ti->sig;
  return sizeof(kaapic_task_info_t)+ts->nargs*sizeof(kaapic_arg_info_t);
}

static void kaapic_taskformat_task_copy(const struct kaapi_format_t* fmt, void* sp_dest, const void* sp_src)
{
  kaapic_task_info_t* const ti_dest = (kaapic_task_info_t*)sp_dest;
  const kaapic_task_info_t* const ti_src = (const kaapic_task_info_t*)sp_src;
  const kaapic_task_sig_t* const ts = ti_src->sig;
  ti_dest->body  = ti_src->body;
  ti_dest->sig   = ti_src->sig;
  for (int i=0; i<ts->nargs; ++i)
    ti_dest->args[i] = ti_src->args[i];
}

static size_t kaapic_taskformat_get_count_params(
 const struct kaapi_format_t* f,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  const kaapic_task_sig_t* const ts = ti->sig;
  return ts->nargs;
}

static kaapi_access_mode_t kaapic_taskformat_get_mode_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  const kaapic_task_sig_t* const ts = ti->sig;
  const kaapi_access_mode_t m = ts->args[i].mode;
  return m;
}

static void* kaapic_taskformat_get_off_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  void* off = (void*)&(ti->args[i].access.data);
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
  const kaapic_task_sig_t* const ts = ti->sig;
  const struct kaapi_format_t* const format = ts->args[i].format;
  return format;
}

static kaapi_memory_view_t kaapic_taskformat_get_view_param(
 const struct kaapi_format_t* f,
 unsigned int i,
 const void* p
)
{
  const kaapic_task_info_t* const ti = p;
  const kaapic_task_sig_t* const ts = ti->sig;
  return ts->args[i].view;
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
  kaapic_task_sig_t* const ts = ti->sig;
  ts->args[i].view = *v;
}

#define _KAAPIC_DECLNAME_REDOP(name,type) _kaapic_redop_##name
#define _KAAPIC_DECLNAME_REDINIT(name,type) _kaapic_redinit_##name

#define _KAAPIC_DECL_REDOP_FLOATS(C, name, type, ...)	\
  _KAAPIC_DECL_REDOP_BASE(C, name, type, , )
#define _KAAPIC_DECL_REDOP_INTEGERS(C, name, type, ...)			\
  _KAAPIC_DECL_REDOP_BASE(C, name, type, _KAAPIC_DECL_REDOP_EXT1, _KAAPIC_DECL_REDOP_EXT2)

#define _KAAPIC_DECL_REDOP_EXT1						\
  case KAAPIC_REDOP_AND  : *r &= *d; break;				\
  case KAAPIC_REDOP_OR   : *r |= *d; break;				\
  case KAAPIC_REDOP_XOR  : *r ^= *d; break;				\

#define _KAAPIC_DECL_REDOP_EXT2						\
  case KAAPIC_REDOP_AND  : *r = ~0; break;				\
  case KAAPIC_REDOP_OR   : *r = 0; break;				\
  case KAAPIC_REDOP_XOR  : *r = 0; break;				\

#define _KAAPIC_DECL_REDOP_BASE(C, name, type, ext1, ext2)			\
  static void _KAAPIC_DECLNAME_REDOP(name,type)( int op, void* p, const void* q) \
  {									\
    type* r = (type*)p;							\
    const type* d = (const type*)q;					\
    switch (op) {							\
    case KAAPIC_REDOP_PLUS : *r += *d; break;				\
    case KAAPIC_REDOP_MUL  : *r *= *d; break;				\
    case KAAPIC_REDOP_MINUS: *r -= *d; break;				\
      ext1								\
    case KAAPIC_REDOP_LAND : *r = *r && *d; break;			\
    case KAAPIC_REDOP_LOR  : *r = *r || *d; break;			\
    case KAAPIC_REDOP_MAX  : *r = (*r < *d ? *d : *r); break;		\
    case KAAPIC_REDOP_MIN  : *r = (*r > *d ? *d : *r); break;		\
    default:								\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");	\
    };									\
  }									\
  static void _KAAPIC_DECLNAME_REDINIT(name,type)( int op, void* p)	\
  {									\
    type* r = (type*)p;							\
    switch (op) {							\
    case KAAPIC_REDOP_PLUS : *r = 0; break;				\
    case KAAPIC_REDOP_MUL  : *r = 1; break;				\
    case KAAPIC_REDOP_MINUS: *r = 0; break;				\
      ext2								\
    case KAAPIC_REDOP_LAND : *r = 1; break;				\
    case KAAPIC_REDOP_LOR  : *r = 0; break;				\
    case KAAPIC_REDOP_MAX  : *r = C##_MIN; break;			\
    case KAAPIC_REDOP_MIN  : *r = C##_MAX; break;			\
    default:								\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");	\
    };									\
  }

_KAAPIC_TYPE_ITER_INTEGERS(_KAAPIC_DECL_REDOP_INTEGERS, )
_KAAPIC_TYPE_ITER_FLOATS(_KAAPIC_DECL_REDOP_FLOATS, )

typedef void (*_kaapic_redop_func_t)(int op, void*, const void*);

_kaapic_redop_func_t all_redops[] = {
#define _KAAPI_m(C,c,type,...) \
  _KAAPIC_DECLNAME_REDOP(c,type)
  _KAAPIC_TYPE_ITER_INTEGERS_COMMA(_KAAPI_m),
  _KAAPIC_TYPE_ITER_FLOATS_COMMA(_KAAPI_m)
#undef _KAAPI_m
};

typedef void (*_kaapic_redinit_func_t)(int op, void*);
_kaapic_redinit_func_t all_redinits[] = {
#define _KAAPI_m(C,c,type,...) \
  _KAAPIC_DECLNAME_REDINIT(c,type)
  _KAAPIC_TYPE_ITER_INTEGERS_COMMA(_KAAPI_m),
  _KAAPIC_TYPE_ITER_FLOATS_COMMA(_KAAPI_m)
#undef _KAAPI_m
};

__attribute__((unused))
static void kaapic_taskformat_reducor
(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* sp,
 const void* q
)
{
  const kaapic_task_info_t* const ti = sp;
  const kaapic_task_sig_t* const ts = ti->sig;
  const kaapic_arg_info_t* argi = &ti->args[i];
  const kaapic_arg_sig_t* args = &ts->args[i];
  kaapi_assert_debug( args->type < KAAPIC_TYPE_PTR );

  (*all_redops[args->type])(args->redop, argi->access.data, q);
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
  const kaapic_task_info_t* const ti = sp;
  const kaapic_task_sig_t* const ts = ti->sig;
  const kaapic_arg_sig_t* args = &ts->args[i];
  kaapi_assert_debug( args->type < KAAPIC_TYPE_PTR );

  (*all_redinits[args->type])(args->redop, p);
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

static kaapi_hashmap_t hash_task_sig;

void _kaapic_register_task_format(void)
{
  struct kaapi_format_t* format = kaapi_format_allocate();
  kaapi_format_taskregister_func
  (
    format,
    kaapic_dfg_body,
    (kaapi_task_body_t)kaapic_dfg_body_wh,
    "kaapic_dfg_task",
    kaapic_taskformat_get_size,
    kaapic_taskformat_task_copy,
    kaapic_taskformat_get_count_params,
    kaapic_taskformat_get_mode_param,
    kaapic_taskformat_get_off_param,
    kaapic_taskformat_get_access_param,
    kaapic_taskformat_set_access_param,
    kaapic_taskformat_get_fmt_param,
    kaapic_taskformat_get_view_param,
    kaapic_taskformat_set_view_param,
    kaapic_taskformat_reducor, /* reducor */
    kaapic_taskformat_redinit, /* redinit */
    0, /* task binding */
    0  /* get_splitter */
  );

  format = kaapi_format_allocate();
  kaapi_format_taskregister_func
  (
    format,
    kaapic_dfg_body_fortran,
    (kaapi_task_body_t)kaapic_dfg_body_wh_fortran,
    "kaapic_dfg_task",
    kaapic_taskformat_get_size,
    kaapic_taskformat_task_copy,
    kaapic_taskformat_get_count_params,
    kaapic_taskformat_get_mode_param,
    kaapic_taskformat_get_off_param,
    kaapic_taskformat_get_access_param,
    kaapic_taskformat_set_access_param,
    kaapic_taskformat_get_fmt_param,
    kaapic_taskformat_get_view_param,
    kaapic_taskformat_set_view_param,
    kaapic_taskformat_reducor, /* reducor */
    kaapic_taskformat_redinit, /* redinit */
    0, /* task binding */
    0  /* get_splitter */
  );

  kaapi_hashmap_init(&hash_task_sig,0);

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

/* ########################################################################
 * ########################################################################
 *                       Spawn function
 * ########################################################################
 * ########################################################################
 */

/* dataflow interface
   New parsing of arguments is :
      - MODE, [REDOP,] TYPE, count, data
   The old one parse the type after the data which impose sever restriction for the
   C API when pass by value argument are floating point value.

   Only one function is used for all various possibility. This function
   has its 'inline_mode' parameters known at compile time, so that all
   _FEATURE(x) can be optimized *at compile time* (at the cost of code
   duplication for all various main entry point) 
*/

/* if present, FORTRAN ABI is used for V parameters (ie pointer to args) */
#define _KAAPIC_SPAWN_FORTRAN_ABI (1<<0)

/* if present, the signature ts is used as is
   if not present, the signature is built
   __kaapic_spawn2 will respect this flag
   __kaapic_spawn look into the hash to (perhaps) add this flag
*/
#define _KAAPIC_SPAWN_SIG_PROVIDED (1<<1)

/* if present and not SIG_PROVIDED, the built signature is put in the hash */
#define _KAAPIC_SPAWN_PUT_SIG_IN_HASH (1<<2)

/* if present and not SIG_PROVIDED, the signature is fully built
   Note: some part of the signature (offset computation of values)
   can be omitted if the built signature is used only once immediately
*/
#define _KAAPIC_SPAWN_KEEP_SIG (1<<3)

/* if present, do the spawn (else, only signature is managed) */
#define _KAAPIC_SPAWN_DO_SPAWN (1<<4)

/* if present, mode (and redop) are read in the varargs
   (if SIG_PROVIDED is present, in debug mode consistency checks are done )
*/
#define _KAAPIC_SPAWN_PARSE_MODE (1<<5)

/* if present, type are read in the varargs
   (if SIG_PROVIDED is present, in debug mode consistency checks are done )
*/
#define _KAAPIC_SPAWN_PARSE_TYPE (1<<6)

/* if present, mode are read in the varargs */
#define _KAAPIC_SPAWN_PARSE_COUNT (1<<7)

/* if present, argument values are read in the varargs */
#define _KAAPIC_SPAWN_PARSE_ARGS (1<<8)

/* if present, libffi use is forced, even if not required (ie only pointers) */
#define _KAAPIC_SPAWN_FORCE_FFI (1<<9)

typedef struct type_info {
  kaapi_format_t* format;
  size_t size;
  int    align;
  int    v_arg_in_access;
  int    use_ffi;
} type_info_t;

#define alignof(x) __alignof__(x)

static kaapi_access_t _access_temp;

static type_info_t _type_info[] = 
{
#define _KAAPI_m(C,c,type,p,r,ffi_type,...)			\
  [KAAPIC_TYPE_##C]={						\
  .format=&kaapi_##c##_format_object,				\
  .size=sizeof(type),						\
  .align=alignof(type),						\
  .v_arg_in_access=						\
  ((sizeof(type) <= sizeof(_access_temp.version))		\
   && (alignof(type) <= alignof(_access_temp.version))),	\
  .use_ffi=_USE_FFI(type),					\
  }

#define _USE_FFI(type) ((sizeof(type)<=sizeof(void*))?0:_KAAPIC_TS_USE_FFI)
  _KAAPIC_TYPE_ITER_INTEGERS_COMMA(_KAAPI_m),
#undef _USE_FFI
#define _USE_FFI(type) _KAAPIC_TS_USE_FFI
  _KAAPIC_TYPE_ITER_FLOATS_COMMA(_KAAPI_m),
#undef _USE_FFI
#define _USE_FFI(type) 0
  _KAAPIC_TYPE_ITER_OTHERS_COMMA(_KAAPI_m)
#undef _USE_FFI
#undef _KAAPI_m
};

static ffi_type *ffi_type_type[] =
{
#define _KAAPI_m(C,c,type,p,r,ffi_type,...)	\
  [KAAPIC_TYPE_ ## C]=&ffi_type_##ffi_type
  _KAAPIC_TYPE_ITER_COMMA(_KAAPI_m)
#undef _KAAPI_m
};

/* ########################################################################
 * ########################################################################
 *                       Main spawn function
 * ########################################################################
 * ########################################################################
 */
static inline int __kaapic_spawn2(
  int inline_mode,
  kaapic_task_sig_t *ts,
  const kaapic_spawn_attr_t* attr,
  int32_t nargs,
  void (*body)(),
  va_list *va_args) __attribute__((always_inline));
static inline int __kaapic_spawn2(
  int inline_mode,
  kaapic_task_sig_t *ts,
  const kaapic_spawn_attr_t* attr,
  int32_t nargs,
  void (*body)(),
  va_list *va_args)
{
#define _FEATURE(name) \
  (inline_mode & _KAAPIC_SPAWN_##name)

  kaapi_thread_t* thread = NULL;
  kaapic_task_info_t* ti;
  void *values;

  if (_FEATURE(DO_SPAWN)) {
    /* thread is only required when really spawning,
       not if we only build the signature
    */
    thread = kaapi_self_thread();
  }

  /* ######################################################################
   *                       Init task signature
   */
  if (_FEATURE(SIG_PROVIDED)) {
    kaapi_assert_debug( ts->nargs == nargs );
  } else {
    size_t taille = sizeof(kaapic_task_sig_t)
	    + nargs*(sizeof(kaapic_arg_sig_t)+sizeof(ffi_type*));
    if (_FEATURE(PUT_SIG_IN_HASH)) {
      ts = malloc(taille);
    } else {
      ts = kaapi_thread_pushdata_align(
	thread, taille, alignof(kaapic_task_sig_t));
    }
    int flags=0;
    if (_FEATURE(FORTRAN_ABI)) {
      flags = _KAAPIC_TS_FORTRAN_ABI;
    }
    /* We need to build the signature, we store some variables */
    if (nargs > KAAPIC_MAX_ARGS)
    {
      /* No shortcut available, FFI forced */
      ts->flags = _KAAPIC_TS_USE_FFI|flags;
    } else {
      ts->flags = flags;
    }
    ts->nargs=nargs;
    ts->values_size=0;
    ts->values_align=0;
  }

  /* ######################################################################
   *                       Init task info
   */
  if (_FEATURE(DO_SPAWN)) {
    ti = kaapi_thread_pushdata_align(
      thread,
      sizeof(kaapic_task_info_t) + nargs*sizeof(kaapic_arg_info_t),
      alignof(kaapic_task_info_t)
    );
    if (_FEATURE(SIG_PROVIDED)) {
      if (ts->values_size) {
	values = kaapi_thread_pushdata_align(
	  thread,
	  ts->values_size,
	  ts->values_align
	);
      } else {
	values=NULL;
      }
    }
    ti->body  = body;
    ti->sig   = ts;
  }

#ifndef __BIGGEST_ALIGNMENT__
#  warning __BIGGEST_ALIGNMENT__ not available, using 16
#  define __BIGGEST_ALIGNMENT__ 16
#endif

  /* ######################################################################
   *                       Arguments loop
   * ######################################################################
   */
  for (unsigned int k = 0; k < nargs; ++k)
  {
    kaapic_arg_info_t* const ai = &ti->args[k];
    kaapic_arg_sig_t* const as = &ts->args[k];

    /* ####################################################################
     *                     Parse mode
     */
    {
      if (_FEATURE(PARSE_MODE)) {
	/* parse arg */
	int modec;
	int modef;
	if(_FEATURE(FORTRAN_ABI)) {
	  modef = *va_arg(*va_args, int *);
	  modec = modef2modec[modef];
	} else {
	  modec = va_arg(*va_args, int);
	}
	if (_FEATURE(SIG_PROVIDED)) {
	  kaapi_assert_debug( modec2modek[modec] == as->mode );
	} else {
	  if(_FEATURE(FORTRAN_ABI)) {
	    if ((modef >= KAAPIF_MODE_MAX) || (modef <0))
	    {
	      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapif_spawn] invalid 'mode' argument\n");)
		return KAAPIF_ERR_EINVAL;
	    }
	  } else {
	    if ((modec >= _KAAPIC_MODE_MAX) || (modec <0))
	    {
	      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid 'mode' argument\n");)
		return EINVAL;
	    }
	  }
	  if (modec == KAAPIC_MODE_T) {
	    ts->flags |= _KAAPIC_TS_HAS_SCRATCH;
	  }
	  as->mode = modec2modek[modec];
	}
      }
    }
    const int mode = as->mode;

    /* ####################################################################
     *                     Parse redop
     */
    if (_FEATURE(PARSE_MODE)) {
      int redop = KAAPIC_REDOP_VOID;
      if (mode == KAAPI_ACCESS_MODE_CW)
      {
	if(_FEATURE(FORTRAN_ABI)) {
	  KAAPI_DEBUG_INST(fprintf(stderr,"[kaapif_spawn] reduction not implemented\n");)
	    return KAAPIF_ERR_UNIMPL;
	}
	redop = va_arg(*va_args, int);
	if (_FEATURE(SIG_PROVIDED)) {
	  kaapi_assert_debug( redop == as->redop );
	} else {
	  if ((redop > KAAPIC_REDOP_MIN) || (redop <= 0))
	  {
	    KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid reduction operator\n");)
	      return EINVAL;
	  }
	}
      }
      if (!_FEATURE(SIG_PROVIDED)) {
	as->redop = redop;
      }
    }

    /* ####################################################################
     *                     Parse type
     */
    if (_FEATURE(PARSE_TYPE)) {
      int typec, typef;
      if(_FEATURE(FORTRAN_ABI)) {
	typef = *va_arg(*va_args, int *);
	typec = typef2typec[typef];
      } else {
	typec = va_arg(*va_args, int);
      }
      if (_FEATURE(SIG_PROVIDED)) {
	kaapi_assert_debug( typec == as->type );
      } else {
	if(_FEATURE(FORTRAN_ABI)) {
	  if ((typef >= KAAPIF_TYPE_MAX) || (typef <0))
	  {
	    KAAPI_DEBUG_INST(fprintf(stderr,"[kaapif_spawn] invalid 'type' argument\n");)
	      return KAAPIF_ERR_EINVAL;
	  }
	} else {
	  if ((typec >= _KAAPIC_TYPE_MAX) || (typec <0))
	  {
	    KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid 'type' argument\n");)
	      return EINVAL;
	  }
	}
	as->type = typec;
      }
    }
    const int type = as->type;
    const type_info_t *typeinfo=&_type_info[type];
    if (!_FEATURE(SIG_PROVIDED)) {
      as->format = typeinfo->format;
    }

    /* ####################################################################
     *                     Parse count
     */
    if (_FEATURE(PARSE_COUNT)) {
      int count;
      if(_FEATURE(FORTRAN_ABI)) {
	count = *va_arg(*va_args, int *);
      } else {
	count = va_arg(*va_args, int);
      }
      if (!_FEATURE(SIG_PROVIDED)) {
	size_t wordsize   = typeinfo->size;
	as->view = kaapi_memory_view_make1d(count, wordsize);
      }
      if (mode == KAAPI_ACCESS_MODE_V) {
	kaapi_assert_debug( count == 1 );
      }
    }

    /* ####################################################################
     *                     compute voffset for V-args
     */
    if (!_FEATURE(SIG_PROVIDED)) {
      if (mode == KAAPI_ACCESS_MODE_V) {
	ts->flags |= typeinfo->use_ffi;
	if (typeinfo->v_arg_in_access) {
	  as->voffset = -1;
	} else {
	  if (_FEATURE(KEEP_SIG)) {
	    size_t align = typeinfo->align;
	    as->voffset = (ts->values_size + (align - 1)) & (align - 1);
	    ts->values_size = as->voffset + typeinfo->size;
	    if (ts->values_align < align) {
	      ts->values_align = align;
	    }
	  } else {
	    /* no need to compute the offset :
	       the signature will not be reused */
	    as->voffset = 0;
	  }
	}
      }
    }

    /* ####################################################################
     *                     Parse argument
     */
    if (_FEATURE(PARSE_ARGS)) {
      void *addr;
      if (mode != KAAPI_ACCESS_MODE_V) {
	addr = va_arg(*va_args, void*);
	kaapi_access_init( &ai->access, addr );
      } else {
	if (as->voffset==-1) {
	  addr = &ai->access.version;
	} else {
	  if (_FEATURE(SIG_PROVIDED)) {
	    /* The signature was available, one block have been allocated */
	    addr = values + as->voffset;
	  } else {
	    /* The value must get some place on the stack */
	    addr = kaapi_thread_pushdata_align(
	      thread, typeinfo->size, typeinfo->align);
	  }
	}
	kaapi_access_init( &ai->access, addr );
	/* kaapi_access_init set access.version to 0
	   so it must be called before the following switch
	   that can but the value into access.version */
	if (_FEATURE(FORTRAN_ABI)) {
          /* FORTRAN pointer is passed, but &ptr must be given to task */
	  switch (type) {
#define _KAAPI_m(C,c,type,...)				\
	    case KAAPIC_TYPE_##C:			\
	      *(type*)addr = *va_arg(*va_args, type*);	\
	      break
	    _KAAPIC_TYPE_ITER_INTEGERS(_KAAPI_m, ;) ;
	    _KAAPIC_TYPE_ITER_FLOATS(_KAAPI_m, ;) ;
#undef _KAAPI_m
#define _KAAPI_m(C,c,type,promtype,...)				\
	    case KAAPIC_TYPE_##C:				\
	      *(type*)addr = va_arg(*va_args, promtype);	\
	      break
	    _KAAPIC_TYPE_ITER_OTHERS(_KAAPI_m, ;) ;
	  default:
	    break;
	  }
	} else {
	  switch (type) {
	    _KAAPIC_TYPE_ITER(_KAAPI_m, ;) ;
#undef _KAAPI_m
	  default:
	    break;
	  }
	}
      }
    }
  }
  /* ######################################################################
   *                       End of arguments loop
   * ######################################################################
   */

  /* ######################################################################
   *                       Compute FFI signature
   */
  if ((!_FEATURE(SIG_PROVIDED))
      && (_FEATURE(FORCE_FFI) || KAAPIC_TS(ts, USE_FFI))) {
    if (_FEATURE(FORCE_FFI)) {
      ts->flags |= _KAAPIC_TS_USE_FFI;
    }

    /* the array of ffi_type* is put after the array of kaapic_arg_sig_t */
    ffi_type **ffi_args=(ffi_type **)&ts->args[nargs];

    for (unsigned int k = 0; k < nargs; ++k) {
      kaapic_arg_sig_t* const as = &ts->args[k];
      if (as->mode != KAAPI_ACCESS_MODE_V) {
	ffi_args[k]=&ffi_type_pointer;
      } else {
	ffi_args[k]=ffi_type_type[as->type];
      }
    }
    if (!ffi_prep_cif(&ts->cif, FFI_DEFAULT_ABI, nargs,
		      &ffi_type_void, ffi_args) == FFI_OK) {
      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] error while initializing ffi\n");)
    }
  }

  /* ######################################################################
   *                       Store task signature
   */
  if ((!_FEATURE(SIG_PROVIDED)) &&(_FEATURE(PUT_SIG_IN_HASH))) {
    kaapi_hashentries_t* entry;
    entry = kaapi_hashmap_findinsert(&hash_task_sig,body);
    if (entry->u.ts !=0) {
      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] several registration of %p\n", body);)
	free(entry->u.ts);
    }
    entry->u.ts = ts;
  }

  /* ######################################################################
   *                       Spawn the task
   */
  if (_FEATURE(DO_SPAWN)) {
    if (_FEATURE(FORTRAN_ABI)) {
      return kaapic_spawn_ti( thread, attr, kaapic_dfg_body_fortran, ti );
    } else {
      return kaapic_spawn_ti( thread, attr, kaapic_dfg_body, ti );
    }
  } else {
    return 0;
  }

#undef _FEATURE
}

static inline int __kaapic_spawn(
  int inline_mode,
  kaapic_task_sig_t *ts,
  const kaapic_spawn_attr_t* attr,
  int32_t nargs,
  void (*body)(),
  va_list *va_args) __attribute__((always_inline));
static inline int __kaapic_spawn(
  int inline_mode,
  kaapic_task_sig_t *ts,
  const kaapic_spawn_attr_t* attr,
  int32_t nargs,
  void (*body)(),
  va_list *va_args)
{
#define _FEATURE(name) \
  (inline_mode & _KAAPIC_SPAWN_##name)

  if (! _FEATURE(SIG_PROVIDED)) {
    kaapi_hashentries_t* entry;
    entry=kaapi_hashmap_find(&hash_task_sig, body);
    if(entry) {
      ts=entry->u.ts;
    }
  }
  if (ts==NULL) {
    return __kaapic_spawn2(inline_mode, NULL,
			   attr, nargs, body, va_args);
  } else {
    return __kaapic_spawn2(inline_mode|_KAAPIC_SPAWN_SIG_PROVIDED, ts,
			   attr, nargs, body, va_args);
  }
#undef _FEATURE
}

int kaapic_spawn(const kaapic_spawn_attr_t* attr, int32_t nargs,
		 void (*body)(),  ...)
{
  int ret;
  va_list va_args;
  va_start(va_args, body);
#define KS(name) _KAAPIC_SPAWN_##name
  ret = __kaapic_spawn(
    0
    |KS(PUT_SIG_IN_HASH)|KS(KEEP_SIG)|KS(DO_SPAWN)
    |KS(PARSE_MODE)|KS(PARSE_TYPE)|KS(PARSE_COUNT)|KS(PARSE_ARGS),
    NULL, attr, nargs, body, &va_args);
#undef KS
  va_end(va_args);
  return ret;
}

int _kaapic_spawn_fortran
(
 int32_t* nargs,
 void (*body)(),
 va_list *va_args
)
{
  int ret;
#define KS(name) _KAAPIC_SPAWN_##name
  ret = __kaapic_spawn(
    KS(FORTRAN_ABI)
    |KS(PUT_SIG_IN_HASH)|KS(KEEP_SIG)|KS(DO_SPAWN)
    |KS(PARSE_MODE)|KS(PARSE_TYPE)|KS(PARSE_COUNT)|KS(PARSE_ARGS),
    NULL, NULL,	*nargs, body, va_args);
#undef KS
  return ret;
}
