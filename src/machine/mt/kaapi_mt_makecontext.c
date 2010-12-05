/*
** kaapi_mt_threadcontext.c
** xkaapi
** 
** Created on Tue Mar 31 15:16:47 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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

#if defined(KAAPI_USE_SETJMP)
#  include <setjmp.h>

#  if defined(__APPLE__) || defined(KAAPI_USE_IPHONEOS) /* DARWING */

#    if defined(__PPC__)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#      define KAAPI_SAVE_SPACE (16)
#      define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[0]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[21] = (unsigned long)(f);

#      define get_sp() \
({ \
  register unsigned long sp asm("r1"); \
  sp; \
})

#    elif defined(__arm__)
/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#      define KAAPI_SAVE_SPACE (16)
#      define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[7]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE) & ~0x1f;\
  ((unsigned long*)jb)[8] = ((unsigned long)(f));

#      define get_sp() \
({ \
  register unsigned long sp asm("r13"); \
  sp; \
})


 /* */
#    elif defined(__i386__) || defined(__x86_64__)

/* update stack pointer of the save context and the return address used by _longjmp
   in order to pass the function to call as it is returned from setjmp.
   The first argument of the user called function is the parameter of _longjmp which
   is always this.
*/
#      define KAAPI_SAVE_SPACE (16+4)/*l ajout de 4+4*nb_arg une fois 
                                 l allignement sur 16 est effectue 
                                 pour l emplacement de l adresse de retour*/
#      define kaapi_update_context( jb, f, sp, bz )\
  ((unsigned long*)jb)[8]  = (((unsigned long)sp)+bz-KAAPI_SAVE_SPACE);\
  ((unsigned long*)jb)[9]  = ((unsigned long*)jb)[8];\
  ((unsigned long*)jb)[10] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[14] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[15] = (unsigned long)0x1f;\
  ((unsigned long*)jb)[17] = (unsigned long)0x37;\
  ((unsigned long*)jb)[12] = (unsigned long)(f);

#      define get_sp() \
({ \
  register unsigned long sp asm("esp"); \
  sp; \
})

#    endif /* PPC and ARM and IA32  */
#  endif /* __APPLE__ || defined(KAAPI_USE_IPHONEOS */
#endif /* KAAPI_USE_SETJMP */


#if 0 /* SHOULD BE REINSERT IN THE COMPILATION PROCESS IF WE WANT TO SUSPEND THREADS */
static void kaapi_trampoline_context( kaapi_thread_context_t * ctxt );
#endif

/*
*/
int kaapi_makecontext( 
  kaapi_processor_t* proc __attribute__((unused)), 
  kaapi_thread_context_t* ctxt __attribute__((unused)), 
  void (*entrypoint)(void*) __attribute__((unused)), 
  void* arg __attribute__((unused))
)
{
#if 0 /* SHOULD BE REINSERT IN THE COMPILATION PROCESS IF WE WANT TO SUSPEND THREADS */
  ctxt->arg = arg;
  ctxt->entrypoint = entrypoint;
  
  /* create a context for a user thread */
#if defined(KAAPI_USE_UCONTEXT)
  kaapi_assert( 0 == getcontext(&ctxt->mcontext) );
  ctxt->mcontext.uc_link = 0;
  ctxt->mcontext.uc_stack.ss_flags = 0;
  ctxt->mcontext.uc_stack.ss_sp    = ctxt->cstackaddr;
  ctxt->mcontext.uc_stack.ss_size  = ctxt->cstacksize;
  makecontext( &ctxt->mcontext, (void (*)())&kaapi_trampoline_context, 1, ctxt );
#elif defined(KAAPI_USE_SETJMP)
  _setjmp(ctxt->mcontext);
  ctxt->cstackaddr=(int *)((unsigned long)ctxt->cstackaddr-((unsigned long)ctxt->cstackaddr % (unsigned long)16)+16);
  ctxt->cstacksize=ctxt->cstacksize-((unsigned long)ctxt->cstacksize % (unsigned long)16);
  kaapi_update_context(ctxt->mcontext, &kaapi_trampoline_context, ((char*)ctxt->cstackaddr), ctxt->cstacksize );
#else 
#  error "not implemented"  
#endif  

#endif
  return 0;
}

#if 0 /* SHOULD BE REINSERT IN THE COMPILATION PROCESS IF WE WANT TO SUSPEND THREADS */

/** This method is only call during the first call to the entrypoint on the context
*/
static void __attribute__((unused)) kaapi_trampoline_context ( kaapi_thread_context_t* shouldbe_ctxt )
{
#if defined(KAAPI_USE_UCONTEXT)
  kaapi_thread_context_t* ctxt __attribute__((unused))= shouldbe_ctxt;
#else
#if defined(__APPLE__) && defined(__PPC__)
  /* should be parameter, but ... ? */
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("r3"); arg0; });

#elif defined(__APPLE__) && (defined(__i386) || defined(__x86_64__))
  /* should be parameter, but ... ? */
  register kaapi_thread_context_t* ctxt;
  __asm__ ("movl %%eax, %0;"
            :"=r"(ctxt)     /* output */
            : /* no input */
             /* no clobbered register */
       );       

  /*some problems with this. But without it doesn't do anything, so it is
   * better*/
#elif defined(KAAPI_USE_IPHONEOS) 
#  if defined(__arm__) 
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("r0"); arg0; });
  ctxt = shouldbe_ctxt;
#  elif defined(__i386__) || defined(__x86_64__) 
  kaapi_thread_context_t* ctxt = ({ register kaapi_thread_context_t* arg0 asm("eax"); arg0; });
#  else
#    warning "error for iPhoneOS"
#  endif
#endif
#endif
#if 0
  (*ctxt->entrypoint)(ctxt->arg);
#endif
}

#endif /* SHOULD BE REINSERT IN THE COMPILATION PROCESS IF WE WANT TO SUSPEND THREADS */
