/*
** kaapi_thread_context.c
** xkaapi
** 
** Created on Tue Mar 31 15:16:47 2009
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
#include "kaapi_impl.h"

#if defined(KAAPI_USE_SETJMP)

#if defined(KAAPI_USE_APPLE) || defined(KAAPI_USE_IPHONEOS) /* DARWING */

#if defined(KAAPI_USE_ARCH_PPC)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16)
#define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[0]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[21] = (unsigned long)(f);

#define get_sp() \
({ \
  register unsigned long sp asm("r1"); \
  sp; \
})

#elif defined(KAAPI_USE_ARCH_ARM)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16)
#define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[7]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[8] = ((unsigned long)(f));

#define get_sp() \
({ \
  register unsigned long sp asm("r13"); \
  sp; \
})


 /* */
#elif defined(KAAPI_USE_ARCH_X86)

/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#define KAAPI_SAVE_SPACE (16+4)/*l ajout de 4+4*nb_arg une fois 
                                 l allignement sur 16 est effectue 
                                 pour l emplacement de l adresse de retour*/
#define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[8]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE);\
  ((unsigned long*)jb)[9]  = ((unsigned long*)jb)[8];\
  ((unsigned long*)jb)[10] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[14] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[15] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[17] = (unsigned long)0x37;\
  ((unsigned long*)jb)[12] = (unsigned long)(f);

#define get_sp() \
({ \
  register unsigned long sp asm("esp"); \
  sp; \
})

#endif /* PPC and ARM and IA32  */
#endif /* KAAPI_USE_APPLE) || defined(KAAPI_USE_IPHONEOS */
#endif /* KAAPI_USE_SETJMP */



static void kaapi_trampoline_context( struct kaapi_thread_context_t* ctxt );


/*
*/
int kaapi_makecontext( kaapi_thread_context_t* ctxt, void (*entrypoint)(void* arg), void* arg )
{
  int err;

  ctxt->arg = arg;
  ctxt->entrypoint = entrypoint;
  
  /* create a context for a user thread */
#if defined(KAAPI_USE_UCONTEXT)
  err = getcontext(&ctxt->mcontext);
  kaapi_assert( err == 0);
  ctxt->mcontext.uc_link = 0;
  ctxt->mcontext.uc_stack.ss_flags = 0;
  ctxt->mcontext.uc_stack.ss_sp    = ctxt->cstackaddr;
  ctxt->mcontext.uc_stack.ss_size  = ctxt->cstacksize;
  makecontext( &ctxt->mcontext, (void (*)(void*))&kaapi_trampoline_context, 1, ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _setjmp(ctxt->mcontext);
  ctxt->cstackaddr=(int *)((unsigned long)ctxt->cstackaddr-((unsigned long)ctxt->cstackaddr % (unsigned long)16)+16);
  ctxt->cstacksize=ctxt->cstacksize-((unsigned long)ctxt->cstacksize % (unsigned long)16);
  kaapi_update_context(ctxt->mcontext, &kaapi_trampoline_context, ((char*)ctxt->cstackaddr), ctxt->cstacksize );
#else 
#  error "not implemented"  
#endif  
  return 0;
}


/**
*/
int kaapi_getcontext( kaapi_thread_processor_t* proc, kaapi_thread_context_t* ctxt )
{
  kaapi_assert_debug( proc == kaapi_sched_get_processor() );

  ctxt->flags        = proc->flags;
  ctxt->dataspecific = proc->dataspecific;

  if (ctxt->flags & KAAPI_CONTEXT_SAVE_KSTACK)
  {
    ctxt->kstack     = proc->kstack;
  }

  if (ctxt->flags & KAAPI_CONTEXT_SAVE_CSTACK)
  {
#if defined(KAAPI_USE_UCONTEXT)
    getcontext( &ctxt->mcontext );
#elif defined(KAAPI_USE_SETJMP)
    _setjmp( ctxt->mcontext );
#endif
  }
  return 0;
}


/**
*/
int kaapi_setcontext( kaapi_thread_processor_t* proc, kaapi_thread_context_t* ctxt )
{
  kaapi_assert_debug( proc == kaapi_sched_get_processor() );

  proc->flags        = ctxt->flags;
  proc->dataspecific = ctxt->dataspecific;

  if (ctxt->flags & KAAPI_CONTEXT_SAVE_KSTACK)
  {
    proc->kstack     = ctxt->kstack;
  }

  if (ctxt->flags & KAAPI_CONTEXT_SAVE_CSTACK)
  {
#if defined(KAAPI_USE_UCONTEXT)
    setcontext( &proc->_ctxt );
#elif defined(KAAPI_USE_SETJMP)
    _longjmp( proc->_ctxt,  (int)(long)ctxt);
#endif
  }
  return 0;
}

/** This method is only call during the first call to the entrypoint on the context
*/
static void kaapi_trampoline_context( kaapi_thread_context_t* should_bectxt );
{
#if defined(KAAPI_USE_DARWIN) && defined(KAAPI_USE_ARCH_PPC)
  /* should be parameter, but ... ? */
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("r3"); arg0; });
#elif  defined(KAAPI_USE_DARWIN) && defined(KAAPI_USE_ARCH_IA32) 
  /* should be parameter, but ... ? */
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("eax"); arg0; });
  
  /*some problems with this. But without it doesn't do anything, so it is
   * better*/
#elif defined(KAAPI_USE_IPHONEOS) 
#  if defined(KAAPI_USE_ARCH_ARM) 
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("r0"); arg0; });
  ctxt = should_th;
#  elif defined(KAAPI_USE_ARCH_IA32) 
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("eax"); arg0; });
#  else
#    warning "error for iPhoneOS"
#  endif
#endif
  (*ctxt->entrypoint)(ctxt->arg);
}
