/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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
#ifndef _KAAPI_TRACE_H_
#define _KAAPI_TRACE_H_ 1

#if defined(__cplusplus)
extern "C" {
#endif

/* Mark that we compile source of the library.
   Only used to avoid to include public definitition of some types.
*/
#define KAAPI_COMPILE_SOURCE 1

#include "config.h"
#include "kaapi_error.h"
#include "kaapi_defs.h"


/* ========================================================================= */
/* */
extern uint64_t kaapi_perf_thread_delayinstate(struct kaapi_processor_t* kproc);

/* for perf_regs access: SHOULD BE 0 and 1 
   All counters have both USER and SYS definition (sys == program that execute the scheduler).
   * KAAPI_PERF_ID_T1 is considered as the T1 (computation time) in the user space
   and as TSCHED, the scheduling time if SYS space. In workstealing litterature it is also named Tidle.
   [ In Kaapi, TIDLE is the time where the thread (kprocessor) is not scheduled on hardware... ]
*/
#define KAAPI_PERF_USER_STATE       0
#define KAAPI_PERF_SCHEDULE_STATE   1

/* return a reference to the idp-th performance counter of the k-processor in the current set of counters */
#define KAAPI_PERF_REG(kproc, idp) ((kproc)->curr_perf_regs[(idp)])

/* return a reference to the idp-th USER performance counter of the k-processor */
#define KAAPI_PERF_REG_USR(kproc, idp) ((kproc)->perf_regs[KAAPI_PERF_USER_STATE][(idp)])

/* return a reference to the idp-th USER performance counter of the k-processor */
#define KAAPI_PERF_REG_SYS(kproc, idp) ((kproc)->perf_regs[KAAPI_PERF_SCHEDULE_STATE][(idp)])

/* return the sum of the idp-th USER and SYS performance counters */
#define KAAPI_PERF_REG_READALL(kproc, idp) (KAAPI_PERF_REG_SYS(kproc, idp)+KAAPI_PERF_REG_USR(kproc, idp))

/* internal */
extern void kaapi_perf_global_init(void);
/* */
extern void kaapi_perf_global_fini(void);
/* */
extern void kaapi_perf_thread_init ( kaapi_processor_t* kproc, int isuser );
/* */
extern void kaapi_perf_thread_fini ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_start ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_stop ( kaapi_processor_t* kproc );
/* */
extern void kaapi_perf_thread_stopswapstart( kaapi_processor_t* kproc, int isuser );
/* */
extern int kaapi_mt_perf_thread_state(kaapi_processor_t* kproc);

#if defined(__cplusplus)
}
#endif

#endif /* */
