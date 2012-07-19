
#include "kaapi_impl.h"

#include "kaapi_affinity.h"

#include "kaapi_tasklist.h"

#include "memory/kaapi_mem.h"
#include "memory/kaapi_mem_data.h"
#include "memory/kaapi_mem_host_map.h"


kaapi_processor_t* kaapi_affinity_get_by_data(
       kaapi_processor_t*   kproc,
       kaapi_taskdescr_t*   td
       )
{
//    int kid_current = kproc->kid;
//    int kid_remote = (kid_current+1)%kaapi_count_kprocessors;
    size_t i;
    kaapi_mem_data_t *kmd;
    void* sp;
    sp = td->task->sp;
    if( td->fmt == NULL )
	return kproc;

    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_host_map_t* local_map = kaapi_get_current_mem_host_map();
    kaapi_mem_asid_t local_asid = kaapi_mem_host_map_get_asid(local_map);
    const size_t count_params = kaapi_format_get_count_params( td->fmt, sp );

    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param( td->fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

	    kaapi_access_t access = kaapi_format_get_access_param(
		    td->fmt, i, sp );
	    kaapi_data_t* data = kaapi_data( kaapi_data_t, &access );
	    if( KAAPI_ACCESS_IS_WRITE(m) ) {
		kaapi_mem_host_map_find_or_insert( host_map,
			(kaapi_mem_addr_t)kaapi_pointer2void(data->ptr),
			&kmd );
		kaapi_assert_debug( kmd !=0 );
		const kaapi_mem_asid_t valid_asid = kaapi_mem_data_get_nondirty_asid( kmd );
		if( ( valid_asid != host_asid ) && ( valid_asid != local_asid ) ) {
#if 0
		    fprintf( stdout, "[%s] kid=%lu td=%p(name=%s) "
			    "src_asid=%lu (kid=%lu) to dest_asid=%lu (kid=%lu)\n",
			    __FUNCTION__,
			    (long unsigned int)kproc->kid, 
			    (void*)td, td->fmt->name,
			    (long unsigned int)local_asid,
			    (long unsigned int)kaapi_mem_asid2kid( local_asid ),
			    (long unsigned int)valid_asid,
			    (long unsigned int)kaapi_mem_asid2kid( valid_asid )
			   );
		    fflush(stdout);
#endif
		    return kaapi_all_kprocessors[kaapi_mem_asid2kid( valid_asid )];
		}
	    }
    }

    return kproc;

//    return NULL;
}

/* TODO: if same function: merge kaapi_sched_stealtasklist.c */
int kaapi_affinity_exec_readylist( 
	kaapi_processor_t* kproc
    )
{
    kaapi_task_t*               tasksteal;
    kaapi_tasklist_t*           master_tasklist;
    kaapi_taskstealready_arg_t* argsteal;
    kaapi_thread_context_t*       thread;
    kaapi_taskdescr_t*		td;

    if( kaapi_readylist_pop( kproc->rtl, &td ) != 0 ) 
	return 0;

    thread = kaapi_self_thread_context();
    /* decrement with the number of thief: one single increment in place of several */
#if defined(TASKLIST_ONEGLOBAL_MASTER) 
    if (td->tasklist->master ==0)
        master_tasklist = td->tasklist;
    else
        master_tasklist = td->tasklist->master;

#if !defined(TASKLIST_REPLY_ONETD)
    /* to synchronize steal operation and the recopy of TD for non master tasklist */
    if (master_tasklist != tasklist)
        KAAPI_ATOMIC_ADD( &tasklist->pending_stealop, (taskdescr_end-taskdescr_beg + blocsize-1)/blocsize );  
#endif

#else
    master_tasklist = tasklist;
    /* explicit synchronization= implicit at the end of each tasklist */
    KAAPI_ATOMIC_ADD( &master_tasklist->count_thief, 1 );
#endif

    argsteal
	= (kaapi_taskstealready_arg_t*)kaapi_thread_pushdata(thread->stack.sfp, sizeof(kaapi_taskstealready_arg_t));
    argsteal->master_tasklist       = master_tasklist;
    argsteal->victim_tasklist       = (td->tasklist == master_tasklist ? 0 : td->tasklist);
#if defined(TASKLIST_REPLY_ONETD)
    argsteal->td                    = td; /* recopy of the pointer, need synchro if range */
#endif
    tasksteal = kaapi_thread_toptask( thread->stack.sfp );
    kaapi_task_init( 
	    tasksteal,
	    kaapi_taskstealready_body,
	    argsteal
	);
    kaapi_thread_pushtask( thread->stack.sfp );

    return 1;
}

