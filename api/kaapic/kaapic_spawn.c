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



/* Traitement des valeurs double:
  - convention d'appel des fonctions => int, double passée différement, traitement possible lors de la construction du spawn
  - appel body/task => user function : ici il faudrait faire un switch en fonction du type pour laisser le compilateur C
  empiler correctement les arguments (...)
  Voir si appel ... des fonctions (passage de va-arg au niveau user (mais pas top).
  Voir si macro possible
*/

static size_t wordsize_type[] = 
{ 
  sizeof(char),           /* KAAPIC_TYPE_CHAR =0*/
  sizeof(short),          /* KAAPIC_TYPE_SHORT */
  sizeof(int),            /* KAAPIC_TYPE_INT */
  sizeof(long),           /* KAAPIC_TYPE_LONG */
  sizeof(unsigned char),  /* KAAPIC_TYPE_UCHAR*/
  sizeof(unsigned short), /* KAAPIC_TYPE_USHORT*/
  sizeof(unsigned int),   /* KAAPIC_TYPE_UINT*/
  sizeof(unsigned long),  /* KAAPIC_TYPE_ULONG*/
  sizeof(float),          /* KAAPIC_TYPE_REAL*/
  sizeof(double),         /* KAAPIC_TYPE_DOUBLE*/
  sizeof(void*)           /* KAAPIC_TYPE_PTR */
};

static const kaapi_format_t* format_type[] = 
{ 
  &kaapi_char_format_object,      /* KAAPIC_TYPE_CHAR =0*/
  &kaapi_short_format_object,     /* KAAPIC_TYPE_SHORT */
  &kaapi_int_format_object,       /* KAAPIC_TYPE_INT */
  &kaapi_long_format_object,      /* KAAPIC_TYPE_LONG */
  &kaapi_uchar_format_object,     /* KAAPIC_TYPE_UCHAR*/
  &kaapi_ushort_format_object,    /* KAAPIC_TYPE_USHORT*/
  &kaapi_uint_format_object,      /* KAAPIC_TYPE_UINT*/
  &kaapi_ulong_format_object,     /* KAAPIC_TYPE_ULONG*/
  &kaapi_float_format_object,     /* KAAPIC_TYPE_REAL*/
  &kaapi_double_format_object,    /* KAAPIC_TYPE_DOUBLE*/
  &kaapi_voidp_format_object      /* KAAPIC_TYPE_PTR */
};

static const kaapi_access_mode_t modec2modek[] =
{
  KAAPI_ACCESS_MODE_R,       /* KAAPIC_MODE_R*/
  KAAPI_ACCESS_MODE_W,       /* KAAPIC_MODE_W */
  KAAPI_ACCESS_MODE_RW,      /* KAAPIC_MODE_RW */
  KAAPI_ACCESS_MODE_CW,      /* KAAPIC_MODE_CW */
  KAAPI_ACCESS_MODE_V,       /* KAAPIC_MODE_V */
  KAAPI_ACCESS_MODE_SCRATCH, /* KAAPIC_MODE_T */
  KAAPI_ACCESS_MODE_STACK    /* KAAPIC_MODE_S */
};

/* extern function, required by kaapif_spawn */
void kaapic_dfg_body(void* p, kaapi_thread_t* t)
{
  kaapic_task_info_t* const ti = (kaapic_task_info_t*)p;
  
#include "kaapic_dfg_switch.h"
  KAAPIC_DFG_SWITCH(ti);
}

/* same as kaapic_dfg_body + process of scratch arguments */
void kaapic_dfg_body_scratch(void* p, kaapi_thread_t* t)
{
  kaapic_task_info_t* const ti = (kaapic_task_info_t*)p;
  int nargs = (int)ti->nargs;
  
  /* process scratch mode */
  int scratch_count        = 0;
  kaapi_processor_t* kproc = 0;
  for (int i=0; i<nargs; ++i)
  {
    if (ti->args[i].u.mode & KAAPI_ACCESS_MODE_T)
    {
      if (kproc == 0) kproc = kaapi_get_current_processor();
      size_t szarg = kaapi_memory_view_size(&ti->args[i].view);
      ti->args[i].access.data = _kaapi_gettemporary_data(kproc, scratch_count, szarg);
      ++scratch_count;
    } 
  }

  kaapic_dfg_body(p, t);
}

static void* get_arg_wh
(const kaapic_task_info_t* ti, unsigned int i)
{
  const kaapic_arg_info_t* const ai = &ti->args[i];

  if (ai->u.mode != KAAPI_ACCESS_MODE_V)
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

/* same as kaapic_dfg_body_wh + process of scratch arguments */
static void kaapic_dfg_body_wh_scratch(void* p, kaapi_thread_t* t, kaapi_task_t* task)
{
  kaapic_task_info_t* const ti = (kaapic_task_info_t*)p;
  int nargs = (int) ti->nargs;
  
  /* process scratch mode */
  int scratch_count        = 0;
  kaapi_processor_t* kproc = 0;
  for (int i=0; i<nargs; ++i)
  {
    if (ti->args[i].u.mode & KAAPI_ACCESS_MODE_T)
    {
      if (kproc == 0) kproc = kaapi_get_current_processor();
      size_t szarg = kaapi_memory_view_size(&ti->args[i].view);
      ti->args[i].access.data = _kaapi_gettemporary_data(kproc, scratch_count, szarg);
      ++scratch_count;
    } 
  }

  kaapic_dfg_body_wh(p, t, task);
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
  const kaapi_access_mode_t m = ti->args[i].u.mode;
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

#define _KAAPIC_DECLNAME_REDOP(name,type) _kaapic_redop_##name
#define _KAAPIC_DECLNAME_REDINIT(name,type) _kaapic_redinit_##name


#define _KAAPIC_DECL_REDOP(name, type, LMAX, LMIN) \
static void _KAAPIC_DECLNAME_REDOP(name,type)( int op, void* p, const void* q) \
{\
  type* r = (type*)p; \
  const type* d = (const type*)q; \
  switch (op) {\
    case KAAPIC_REDOP_PLUS : *r += *d; break; \
    case KAAPIC_REDOP_MUL  : *r *= *d; break; \
    case KAAPIC_REDOP_MINUS: *r -= *d; break; \
    case KAAPIC_REDOP_AND  : *r &= *d; break; \
    case KAAPIC_REDOP_OR   : *r |= *d; break; \
    case KAAPIC_REDOP_XOR  : *r ^= *d; break; \
    case KAAPIC_REDOP_LAND : *r = *r && *d; break; \
    case KAAPIC_REDOP_LOR  : *r = *r || *d; break; \
    case KAAPIC_REDOP_MAX  : *r = (*r < *d ? *d : *r); break; \
    case KAAPIC_REDOP_MIN  : *r = (*r > *d ? *d : *r); break; \
    default:\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");\
  };\
}\
static void _KAAPIC_DECLNAME_REDINIT(name,type)( int op, void* p)\
{\
  type* r = (type*)p; \
  switch (op) {\
    case KAAPIC_REDOP_PLUS : *r = 0; break; \
    case KAAPIC_REDOP_MUL  : *r = 1; break; \
    case KAAPIC_REDOP_MINUS: *r = 0; break; \
    case KAAPIC_REDOP_AND  : *r = ~0; break; \
    case KAAPIC_REDOP_OR   : *r = 0; break; \
    case KAAPIC_REDOP_XOR  : *r = 0; break; \
    case KAAPIC_REDOP_LAND : *r = 1; break; \
    case KAAPIC_REDOP_LOR  : *r = 0; break; \
    case KAAPIC_REDOP_MAX  : *r = LMIN; break; \
    case KAAPIC_REDOP_MIN  : *r = LMAX; break; \
    default:\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");\
  };\
}

#define _KAAPIC_DECL_REDOPF(name, type, LMAX, LMIN) \
static void _KAAPIC_DECLNAME_REDOP(name,type)( int op, void*p, const void* q) \
{\
  type* r = (type*)p; \
  const type* d = (const type*)q; \
  switch (op) {\
    case KAAPIC_REDOP_PLUS : *r += *d; break; \
    case KAAPIC_REDOP_MUL  : *r *= *d; break; \
    case KAAPIC_REDOP_MINUS: *r -= *d; break; \
    case KAAPIC_REDOP_LAND : *r = *r && *d; break; \
    case KAAPIC_REDOP_LOR  : *r = *r || *d; break; \
    case KAAPIC_REDOP_MAX  : *r = (*r < *d ? *d : *r); break; \
    case KAAPIC_REDOP_MIN  : *r = (*r > *d ? *d : *r); break; \
    default:\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");\
  };\
}\
static void _KAAPIC_DECLNAME_REDINIT(name,type)( int op, void* p)\
{\
  type* r = (type*)p; \
  switch (op) {\
    case KAAPIC_REDOP_PLUS : *r = 0; break; \
    case KAAPIC_REDOP_MUL  : *r = 1; break; \
    case KAAPIC_REDOP_MINUS: *r = 0; break; \
    case KAAPIC_REDOP_LAND : *r = 1; break; \
    case KAAPIC_REDOP_LOR  : *r = 0; break; \
    case KAAPIC_REDOP_MAX  : *r = LMIN; break; \
    case KAAPIC_REDOP_MIN  : *r = LMAX; break; \
    default:\
      kaapi_assert_m(0, "[kaapic]: invalid reduction operator");\
  };\
}

_KAAPIC_DECL_REDOP(char,char,CHAR_MAX,CHAR_MIN)
_KAAPIC_DECL_REDOP(short,short,SHRT_MAX, SHRT_MIN)
_KAAPIC_DECL_REDOP(int,int,INT_MAX, INT_MIN)
_KAAPIC_DECL_REDOP(long,long,LONG_MAX, LONG_MIN)
_KAAPIC_DECL_REDOP(uchar,unsigned char,UCHAR_MAX, 0)
_KAAPIC_DECL_REDOP(ushort,unsigned short, USHRT_MAX, 0)
_KAAPIC_DECL_REDOP(uint,unsigned int, UINT_MAX, 0)
_KAAPIC_DECL_REDOP(ulong,unsigned long, ULONG_MAX, 0)
_KAAPIC_DECL_REDOPF(float,float, FLT_MAX, FLT_MIN)
_KAAPIC_DECL_REDOPF(double,double, DBL_MAX, DBL_MIN)

typedef void (*_kaapic_redop_func_t)(int op, void*, const void*);

_kaapic_redop_func_t all_redops[] = {
  _KAAPIC_DECLNAME_REDOP(char,char),
  _KAAPIC_DECLNAME_REDOP(short,short),
  _KAAPIC_DECLNAME_REDOP(int,int),
  _KAAPIC_DECLNAME_REDOP(long,long),
  _KAAPIC_DECLNAME_REDOP(uchar,unsigned char),
  _KAAPIC_DECLNAME_REDOP(ushort,unsigned short),
  _KAAPIC_DECLNAME_REDOP(uint,unsigned int),
  _KAAPIC_DECLNAME_REDOP(ulong,unsigned long),
  _KAAPIC_DECLNAME_REDOP(float,float),
  _KAAPIC_DECLNAME_REDOP(double,double)  
};

typedef void (*_kaapic_redinit_func_t)(int op, void*);
_kaapic_redinit_func_t all_redinits[] = {
  _KAAPIC_DECLNAME_REDINIT(char,char),
  _KAAPIC_DECLNAME_REDINIT(short,short),
  _KAAPIC_DECLNAME_REDINIT(int,int),
  _KAAPIC_DECLNAME_REDINIT(long,long),
  _KAAPIC_DECLNAME_REDINIT(uchar,unsigned char),
  _KAAPIC_DECLNAME_REDINIT(ushort,unsigned short),
  _KAAPIC_DECLNAME_REDINIT(uint,unsigned int),
  _KAAPIC_DECLNAME_REDINIT(ulong,unsigned long),
  _KAAPIC_DECLNAME_REDINIT(float,float),
  _KAAPIC_DECLNAME_REDINIT(double,double)  
};

__attribute__((unused)) 
static void kaapic_taskformat_reducor
(
 const struct kaapi_format_t* f,
 unsigned int i,
 void* p,
 const void* q
)
{
  const kaapic_task_info_t* const ti = p;
  const kaapic_arg_info_t* argi = &ti->args[i];
  kaapi_assert_debug( argi->u.type <= KAAPIC_TYPE_DOUBLE );
  
  (*all_redops[argi->u.type])(argi->u.redop, p, q);
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
  const kaapic_task_info_t* const ti = p;
  const kaapic_arg_info_t* argi = &ti->args[i];
  kaapi_assert_debug( argi->u.type <= KAAPIC_TYPE_DOUBLE );
  
  (*all_redinits[argi->u.type])(argi->u.redop, p);
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

  format = kaapi_format_allocate();
  kaapi_format_taskregister_func
  (
    format,
    kaapic_dfg_body_scratch, 
    (kaapi_task_body_t)kaapic_dfg_body_wh_scratch,
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

/* dataflow interface 
   New parsing of arguments is :
      - MODE, TYPE, count, data
   The old one parse the type after the data which impose sever restriction for the
   C API when pass by value argument are floating point value.
   
   Because the task body call the user entry points, all access modes are the following:
   -    
*/
int kaapic_spawn(const kaapic_spawn_attr_t* attr, int32_t nargs, ...)
{
  kaapi_thread_t* thread = kaapi_self_thread();
  kaapic_task_info_t* ti;
  va_list va_args;
  size_t wordsize;
  unsigned int k;
  int redop;
  void* addr;
  union {
    int  i;
    unsigned int ui;
    long l;
    unsigned long ul;
    double d;
    void* p;
    uintptr_t uip;
  } value;

  int scratch_arg = 0;

  if (nargs > KAAPIC_MAX_ARGS)
  {
    KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] to many arguments\n");)
    return EINVAL;
  }

  va_start(va_args, nargs);
    
  ti = kaapi_thread_pushdata_align(
    thread, sizeof(kaapic_task_info_t)+nargs*sizeof(kaapic_arg_info_t), sizeof(void*)
  );
  ti->body = va_arg(va_args, void (*)());
  ti->nargs = nargs;

  for (k = 0; k < nargs; ++k)
  {
    value.uip = 0UL;
    redop = KAAPIC_REDOP_VOID;
    kaapic_arg_info_t* const ai = &ti->args[k];

    /* parse arg */
    const int mode  = va_arg(va_args, int);
    if ((mode > KAAPIC_MODE_S) || (mode <0))
    {
      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid 'mode' argument\n");)
      return EINVAL;
    }
    
    if (mode == KAAPIC_MODE_CW)
    {
      redop = va_arg(va_args, int);
      if ((redop >KAAPIC_REDOP_MIN) || (redop <=0))
      {
        KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid reduction operator\n");)
        return EINVAL;
      }
    }

    const int type  = va_arg(va_args, int);
    if ((type >= KAAPIC_TYPE_ID) || (type <0))
    {
      KAAPI_DEBUG_INST(fprintf(stderr,"[kaapic_spawn] invalid 'type' argument\n");)
      return EINVAL;
    }

    const int count = va_arg(va_args, int);


    if (mode != KAAPIC_MODE_V)
      addr = va_arg(va_args, void*);
    else
    {
      switch (type) {
        case KAAPIC_TYPE_SHORT:
        case KAAPIC_TYPE_CHAR:
        case KAAPIC_TYPE_INT:
          value.i  = va_arg(va_args, int); 
          break;
        case KAAPIC_TYPE_LONG:
          value.l = va_arg(va_args, long); 
          break;
        case KAAPIC_TYPE_UCHAR:
        case KAAPIC_TYPE_USHORT:
        case KAAPIC_TYPE_UINT:
          value.ui = va_arg(va_args, unsigned int); 
          break;
        case KAAPIC_TYPE_ULONG:
          value.ul = va_arg(va_args, unsigned long);
          break;
        case KAAPIC_TYPE_FLOAT: /* the callee will receive a pointer to the value for double value */
        case KAAPIC_TYPE_DOUBLE:
          value.d = va_arg(va_args, double);
          break;
        case KAAPIC_TYPE_PTR:
        case KAAPIC_TYPE_ID:
          value.p = (void*)va_arg(va_args, void*); break;
        default:
          break;
      }
    }
    
    ai->u.mode   = modec2modek[mode];
    ai->u.type   = type;
    ai->u.redop  = redop;
    wordsize   = wordsize_type[type];
    ai->format = format_type[type];
    
    if (mode == KAAPIC_MODE_V)
    {
      /* can only pass exactly by value the size of a uintptr_t 
         - should be extended to recopy into the thread stack
      */
      kaapi_assert_debug( wordsize*count <= sizeof(uintptr_t) );
      addr = &ai->access.version; 
      kaapi_access_init( &ai->access, addr );
      memcpy(addr, &value, wordsize*count );  /* but count == 1 here */
    }
    else {
      kaapi_access_init( &ai->access, addr );
      if (mode == KAAPIC_MODE_T)
        scratch_arg = 1;
    }

    ai->view = kaapi_memory_view_make1d(count, wordsize);
  }
  va_end(va_args);

  /* spawn the task */
  if (scratch_arg ==1)
    return kaapic_spawn_ti( thread, attr, kaapic_dfg_body_scratch, ti );
  else
    return kaapic_spawn_ti( thread, attr, kaapic_dfg_body, ti );
}
