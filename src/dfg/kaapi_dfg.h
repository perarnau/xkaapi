/*
** kaapi_dfg.h
** ckaapi
** 
** Created on Tue Mar 31 15:22:24 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#ifndef _KAAPI_DFG_H_
#define _KAAPI_DFG_H_

#include "kaapi_stealapi.h" 
#include <stddef.h>

#define KAAPI_MAX_CLO_PARAMETERS 16

struct kaapi_dfg_closure_t;
struct kaapi_dfg_frame_t;
struct kaapi_access_t;

/** A frame is a FIFO queue of closures that could be put in a stack of frames.
    _next field is the link when a frame is pushed into the stack.
*/
typedef struct kaapi_dfg_frame_t {
  struct kaapi_dfg_frame_t* _next;
  KAAPI_FIFO_DECLARE_FIELD(struct kaapi_dfg_closure_t);
} kaapi_dfg_frame_t;


/** A stack is a LIFO queue of frame with respect to the owner thread.
    The stack is iterated using FIFO order during a steal operation.
*/
typedef struct kaapi_dfg_stack_t {
  kaapi_steal_context_t*      _sc;
  KAAPI_STACK_DECLARE_FIELD( struct kaapi_dfg_frame_t );
} kaapi_dfg_stack_t;


/** init a frame
*/
#define KAAPI_DFG_FRAME_INIT( f ) \
  KAAPI_FIFO_CLEAR(f)

/** init a stack
*/
#define KAAPI_DFG_STACK_INIT( s ) \
  KAAPI_STACK_CLEAR(s)
  
/** init a stack with a given steal context
*/
#define KAAPI_DFG_STACK_INIT_WITH_SC( s, sc ) \
  (s)->_sc = sc;\
  KAAPI_STACK_CLEAR(s)

/** push a new frame after a previous frame 
*/
#define KAAPI_DFG_STACK_PUSH( s, top_saved, f ) \
    top_saved = (s)->_front; \
    if (top_saved !=0) (top_saved)->_next = f;\
    (s)->_front = f

/** pop the frame after oldf 
*/
#define KAAPI_DFG_STACK_POP( s, top_saved ) \
    if (top_saved !=0) (top_saved)->_next = 0; \
    (s)->_front = top_saved

/** Iterate through the stack 
*/
#define KAAPI_DFG_STACK_BACK( s) \
  KAAPI_STACK_BACK(s)

/** Return the next frame of curr
*/
#define KAAPI_DFG_STACK_NEXT_FRAME(s, curr) \
  (curr)->_next;

/** push a closure clo into the frame f 
  kaapi_writemem_barrier();\
*/
#define KAAPI_DFG_FRAME_PUSH( f, clo) \
  { struct kaapi_dfg_closure_t* c = (struct kaapi_dfg_closure_t*)clo;\
    c->_state = 0;\
    kaapi_writemem_barrier();\
    KAAPI_FIFO_PUSH(f, c);\
  }
  
/** kaapi closure entrypoint
*/
typedef void (*kaapi_entry_closure_t)(kaapi_dfg_stack_t*, struct kaapi_dfg_closure_t*);

/** A method to reify a closure
*/
typedef void (*kaapi_reificator_closure_t) ( struct kaapi_dfg_closure_t*);

/** Offset to access to parameter of a closure
*/
typedef unsigned int kaapi_offset_t;

/** Access mode
*/
typedef enum kaapi_access_mode_t {
  KAAPI_ACCESS_INIT     = 0,        /* 0000 0000 : */
  KAAPI_ACCESS_READY    = 1,        /* 0000 0001 : */
  KAAPI_ACCESS_UPDATE   = 2,        /* 0000 0010 : */
  KAAPI_ACCESS_MODE_VOID= 0,        /* 0000 0000 : */
  KAAPI_ACCESS_MODE_V   = 0,        /* 0000 0000 : */
  KAAPI_ACCESS_MODE_R   = 1 << 4,   /* 0001 0000 : */
  KAAPI_ACCESS_MODE_W   = 2 << 4,   /* 0010 0000 : */
  KAAPI_ACCESS_MODE_CW  = 6 << 4,   /* 0110 0000 : */
  KAAPI_ACCESS_MODE_P   = 8 << 4    /* 1000 0000 : */
} kaapi_access_mode_t;

#define KAAPI_ACCESS_MASK_MODE   0xF0
#define KAAPI_ACCESS_MASK_MODE_R 0x10
#define KAAPI_ACCESS_MASK_MODE_W 0x20
#define KAAPI_ACCESS_MASK_MODE_P 0x40

#define KAAPI_ACCESS_GET_MODE( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE )

#define KAAPI_ACCESS_IS_READ( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_R)

#define KAAPI_ACCESS_IS_WRITE( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_W)

#define KAAPI_ACCESS_IS_POSTPONED( m ) \
  ((m) & KAAPI_ACCESS_MASK_MODE_P)

#define KAAPI_ACCESS_IS_ONLYWRITE( m ) \
  (((m) & KAAPI_ACCESS_MASK_MODE_W) && !((m) & KAAPI_ACCESS_MASK_MODE_R))


/** Kaapi format 
*/
typedef struct kaapi_dfg_format_closure_t {
  short                      fmtid;
  short                      isinit;
  const char*                name;
  kaapi_entry_closure_t      entrypoint;
  kaapi_entry_closure_t      termpoint;
  kaapi_reificator_closure_t reificator;
  const kaapi_access_mode_t  mode_params[KAAPI_MAX_CLO_PARAMETERS];   /* only consider value with mask 0xF0 */
  const kaapi_offset_t       params[KAAPI_MAX_CLO_PARAMETERS];        /* access to the i-th parameter: a value or a shared */
  const size_t               size_params[KAAPI_MAX_CLO_PARAMETERS];   /* size of each params */
  const size_t               size_allparams;                           /* size of all params */
  const size_t               size_allaccess;                           /* size of all params of mode access */
  int                        count_params;     /* */
} kaapi_dfg_format_closure_t;

/** All formats of closure 
*/
kaapi_dfg_format_closure_t* kaapi_all_dfg_format_closure[256];

/** Returns the i-th access mode of the closure with format f
*/ 
#define KAAPI_FORMAT_GET_MODE( f, i) \
  ( KAAPI_ACCESS_GET_MODE((f)->mode_params[i]) )

/** Returns the i-th access of the closure c with format f
*/ 
#define KAAPI_FORMAT_GET_ACCESS( f, c, i) \
  ((kaapi_access_t*)( (char*)c + (f)->params[i]) )

/** Returns the i-th shared data of the closure c with format f
*/ 
#define KAAPI_FORMAT_GET_SHARED( f, c, i) \
  (*((void***)( (char*)c + (f)->params[i])) )

/** This is field required for a kaapi closure 
*/
#define KAAPI_DECL_CLOSURE_FIELDS  \
  int                   _state;\
  short                 _flag;\
  short                 _format;\
  kaapi_entry_closure_t _entry;\
  KAAPI_FIFO_FIELD( struct kaapi_dfg_closure_t )

/** This is the kaapi_dfg_closure
*/
typedef struct kaapi_dfg_closure_t {
  KAAPI_DECL_CLOSURE_FIELDS;
} kaapi_dfg_closure_t;

/** Mask for reified _flag
*/
#define KAAPI_MASK_CLOSUREREFIFIED 0x1 /* in flag field */

/** Value for _state field of closure
*/
typedef enum kaapi_dfg_closure_state_t {
  KAAPI_CLOSURE_INIT  = 0,
  KAAPI_CLOSURE_EXEC  = 1,
  KAAPI_CLOSURE_STEAL = 2,
  KAAPI_CLOSURE_TERM  = 3
} kaapi_dfg_closure_state_t;

#define KAAPI_DFG_CLOSURE_GETFORMAT( c ) \
  (kaapi_all_dfg_format_closure[c->_format])

#define KAAPI_DFG_CLOSURE_GETSTATE( c ) \
  ((c)->_state)

#define KAAPI_DFG_CLOSURE_SETSTATE( c, s ) \
  ((c)->_state = s)

#define KAAPI_DFG_CLOSURE_ISINIT( c ) \
  ((c)->_state)

#define KAAPI_DFG_CLOSURE_ISREIFIED( c ) \
  ((c)->_flag & KAAPI_MASK_CLOSUREREFIFIED)

#define KAAPI_DFG_CLOSURE_SETREIFIED( c ) \
  ((c)->_flag |= KAAPI_MASK_CLOSUREREFIFIED)

/** Init a closure
  (clo)->_entry  = (kaapi_entry_closure_t)fnc;\
*/
#define KAAPI_CLOSURE_INIT( clo, fmt ) \
  (clo)->_format = (fmt)->fmtid;\
  (clo)->_entry  = (fmt)->entrypoint;

/** Execute the entrypoint for the closure
    If the function entrypoint is known, it is much faster to use KAAPI_CLOSURE_EXECUTE_FE.
*/
#define KAAPI_CLOSURE_EXECUTE( s, f, clo ) \
  if (KAAPI_DFG_CLOSURE_ISINIT(clo)) kaapi_dfg_suspend( s, f, (kaapi_dfg_closure_t*)clo ); \
  {\
    (clo)->_state = KAAPI_CLOSURE_EXEC;\
    (*KAAPI_DFG_CLOSURE_GETFORMAT(clo)->entrypoint)(s, (kaapi_dfg_closure_t*)clo);\
  }

#define KAAPI_CLOSURE_EXECUTE_FE( s, f, clo, fentrypoint ) \
  if (KAAPI_DFG_CLOSURE_ISINIT(clo)) kaapi_dfg_suspend( s, f, (kaapi_dfg_closure_t*)clo ); \
  {\
    (clo)->_state = KAAPI_CLOSURE_EXEC;\
    fentrypoint(s, (kaapi_dfg_closure_t*)clo);\
  }

#define KAAPI_CLOSURE_EXECUTE_FROM_THIEF( s, f, clo ) \
  (*KAAPI_DFG_CLOSURE_GETFORMAT(clo)->entrypoint)(s, (kaapi_dfg_closure_t*)clo)


/** Kaapi access
    An access is part of the HeavyClosure associated to a closure. An HClosure is only created during steal operation
    in order to build the data flow of a list of LightClosure.
    An access is more or less the access data structure of C++ Kaapi-v0.
*/
typedef struct kaapi_access_t {
  struct kaapi_access_t* __attribute__((aligned)) _next;          /* next acess */
  unsigned short         _status;        /* bit0: data is produced, bit1: _version has been produced */
  unsigned short         _offset_clo;    /* offset to closure base address */
  void*                  _version;       /* the new data if bit0 is set */
} kaapi_access_t;


/** Declare a shared variable of given type in the program
*/
#define KAAPI_DFG_DECLARE_SHARED(type, varname ) \
    type  varname##_data; \
    type* varname##_gd = &varname##_data;\
    type** varname = &varname##_gd

#define KAAPI_SHARED_ACCESS(type, varname ) \
    type** varname;\
    kaapi_access_t varname##_hr
    
#define KAAPI_DFG_READ_SHARED(var ) \
  **var

#define KAAPI_DFG_WRITE_SHARED(var, value ) \
  **var = value

/** Link formal parameter to effective parameter
*/
#define KAAPI_DFG_ACCESS_LINK_VALUE( formal, effective ) \
  formal = effective 

#define KAAPI_DFG_ACCESS_LINK_ACCESS( formal, effective ) \
  formal = effective; \
  formal##_hr._next = 0

#define KAAPI_DFG_ACCESS_NEW_VERSION( a ) (((a)->_status) & 0x2 !=0)

/**
*/
#define KAAPI_DFG_ACCESS_ISREADY( a, m ) ( KAAPI_ACCESS_IS_WRITE(m) || (KAAPI_ACCESS_IS_READ(m) && ((a)->_status) & 0x1 !=0) )

/** Return a pointer to the pointer to the shared object of the closure
*/
#define KAAPI_DFG_ACCESS_GET_CLO(a) \
  ((kaapi_dfg_closure_t*)( (char*)a - _offset_clo))

/** Return a pointer to the pointer to the shared object of the closure
*/
#define KAAPI_DFG_ACCESS_GET_MODE(a, clo) ((void**)

/** Anormal execution of a closure due to no possiblile fast sequential execution
    A call to this function suspended the running user thread untii the closure state is TERM.
*/
void kaapi_dfg_suspend( kaapi_dfg_stack_t* s, kaapi_dfg_frame_t* f, kaapi_dfg_closure_t* c );

/** Build the heavy representation of the closure
*/
void kaapi_dfg_reify_lighclosure( kaapi_dfg_closure_t* c );

/** Return !=0 if closure is ready
*/
int kaapi_dfg_closure_isready( kaapi_dfg_closure_t* c );

/** Compute ready of closure, update link in dfg and return !=0 if closure is ready 
*/
int kaapi_dfg_compute_readyness( kaapi_dfg_stack_t* s, kaapi_dfg_closure_t* c );

/** splitter_work is called within the context of the steal point
*/
void kaapi_dfg_stealercode( 
  kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request
);


/** Entry point to steal task from other threads
*/
void kaapi_dfg_thiefcode(
  kaapi_dfg_stack_t* s, kaapi_dfg_closure_t* c 
);

/**
*/
void kaapi_dfg_thief_entrypoint(kaapi_steal_context_t* sc, void* data);


/** Reify the closure representation
*/
void kaapi_dfg_reify_closure( kaapi_dfg_closure_t* c );

/**
*/
void kaapi_dfg_update_access( kaapi_access_t* a, kaapi_dfg_closure_t* clo, int i );

extern unsigned long long kaapi_timewaiting;



/** Sync
*/
static inline void kaapi_dfg_sync( kaapi_dfg_stack_t* s, kaapi_dfg_frame_t* f )
{
  kaapi_dfg_closure_t* c;
  KAAPI_FIFO_TOP(f,c);
  while (c !=0)
  {
    KAAPI_CLOSURE_EXECUTE(s, f, c);
    c = c->_next;
  }
}


/**
*/
int kaapi_save_dfg( int savemode, int n );
#endif
