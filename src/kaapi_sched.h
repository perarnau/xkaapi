
#ifndef _KAAPI_SCHED_H_
#define _KAAPI_SCHED_H_

/** Used to steal a tasklist in the frame
*/
extern void kaapi_sched_stealtasklist( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_tasklist_t*             tasklist, 
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
);

/** Used to steal a tasklist from local readylist of victim
*/
extern void kaapi_sched_stealreadytasklist( 
                           kaapi_thread_context_t*       thread, 
                           kaapi_readytasklist_t*        rtl, 
                           kaapi_listrequest_t*          lrequests, 
                           kaapi_listrequest_iterator_t* lrrange
);


/** Start execution with a TD.
    Initialize the stack of the current thread and then, the caller shoud calls kaapi_thread_execframe_tasklist.
*/
extern int kaapi_thread_startexecwithtd( 
      kaapi_processor_t* kproc,
      kaapi_taskdescr_t* td
);

int kaapi_thread_execframe_tasklist( kaapi_thread_context_t* thread );

#endif /* _KAAPI_SCHED_H_ */
